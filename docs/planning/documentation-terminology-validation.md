# Documentation terminology and comprehension validation

## Scope and interpretation

Date: 2026-11-04. This record closes the [terminology and comprehension packet](documentation-terminology.md),
following the [earlier didactic redesign](documentation-redesign-validation.md). The whole editorial
baseline is `2906615da17204b8166a827a82b352bd4f13e5f6`. The later
`e6cd5bc7713b596b6d8a196b5b96fa1bcd7519ec` snapshot is a preservation/comparison checkpoint,
not permission to exclude defects introduced by the first terminology pass.

The teaching and independent review covered **978 existing live Markdown pages**, including two
unchanged historical proof README files and the terminology packet itself. This additional validation
record makes the live corpus **979 pages**. The selected source distribution contains **977 Markdown
pages**: evidence directories, including the two proof README files, are deliberately excluded.
[First terms](../terms.md#reading-preserved-evidence) explains that boundary and supplies their
terminology prerequisites without rewriting the proofs.

This is a documentation-only change. It does not add a backend capability, change numerical or
ownership rules, authorize publication, or qualify a new runtime binary. The four sealed
`0.1.0-public-audit.20261104` artifacts and three earlier documentation qualification archives remain
unchanged. The requested autotuning, live CLI and PyTorch performance work is separate from this
editorial qualification; architecture-changing runtime repairs require analysis/proposals rather
than an implicit contract change.

## What a reader can now follow

- [First terms](../terms.md) starts from concrete questions rather than an unexplained acronym list.
  The [dictionary](../glossary.md) introduces prerequisites, examples and distinctions between
  overloaded words; links target the meaning used in the surrounding sentence.
- The [first model](../tutorials/first-model.md#who-supplies-the-weight-and-who-executes-the-rule)
  identifies the application calling Model, the called API, the supplied initialization weight and
  Model's private working copy. A graph **leaf** is a producerless Tensor description, not
  necessarily stored data. [Capture](../reference/compiler/capture.md#why-a-leaf-is-not-necessarily-an-array)
  separates structural graph inputs from the ordinary caller-bindable input list and explicit
  compiler constants.
- [Ownership](../explanation/ownership.md) distinguishes copied values, borrowed inputs and result
  leases. Ordinary Engine input storage remains usable through `RunResult.close`, not merely until
  synchronous execution returns; Model's detached compute result has a different lifetime.
- [The worked objective and gradient](../tutorials/loss-and-gradients.md#why-one-scalar-objective)
  define symbols, domains and the update calculation in LaTeX. Geometry uses checked integer
  reasoning; ideal real-number explanations do not authorize floating-point reassociation or
  broaden non-finite/empty-domain support.
- [Fixed recurrent scans](../reference/tensor/fixed-scans.md#fixed-recurrent-scan-expressions) specify
  GRU reset/update/candidate and LSTM input/forget/candidate/output parameter-block orders.
  [General Scan](../reference/tensor/general-scan.md) and
  [cumulative operations](../reference/tensor/fixed-scans.md#cumulative-scan-expressions) retain their
  different meanings. The [recurrent forward/reverse walkthrough](../architecture/general-differentiable-scan.md#conceptual-forward-and-reverse-example)
  is the prerequisite for sensitivity passing through successive states.
- [Saving and recovery](../how-to/save-and-restart-training.md) distinguish external input values
  from eligible named state, saved source/current values from prepared execution, and model
  reconstruction from exact training continuation.
- The [framework walkthrough](../explanation/framework.md), focused references and package/native
  atlas connect related subjects directly. Current guides use local subsections, short paragraphs,
  meaningful lists and why-comments inside instructional code. Historical records have separated
  reading keys; their original assertions and evidence are not rewritten as current guarantees.
- The root [GitHub README](../../README.md) supplies goal-based entry points, concrete call/ownership
  context, prerequisites, actual support boundaries and the private-distribution restriction.

## Exercised checks

| Surface | Observed result | What it does not establish |
|---|---|---|
| Dictionary | 398 third-level entries; 606 precise `term-` definition anchors | A count alone is not reader comprehension |
| Incoming navigation | All 18,171 original heading/explicit anchors retained | An external URL being online |
| Mathematical notation | 2,305 formulas: 80 display and 2,225 inline; strict KaTeX 0.18.7 HTML plus MathML generation; actual Chromium light/dark rendering without errors or page/display overflow | A new numerical backend qualification |
| Diagrams | All 37 render in light/dark; accessible titles/descriptions; no horizontal overflow; final diagram bodies exactly match the rendered sources | A new runtime execution trace |
| Java examples | All 870 before/after Java fences have identical ordered executable tokens under the real JDK 26 lexer; 214 have comment/whitespace changes; no lexical errors | Compilation of every conceptual fragment |
| Complete copyable programs | All 55 current complete Java classes compiled against the exact 18-JAR installed private reference closure; all 53 classes with `main` actually executed successfully | Re-running the unchanged framework unit suite or all backend arithmetic tests |
| Example/source binding | Final complete-example code exactly matches those compiled/executed sources | Using checkout classes as an external consumer |
| Preserved history | Original visible blocks of 750 historical records accounted for; commands, outputs and original historical formulas retained | Re-proving or rerunning historical measurements |
| Authority and legal boundaries | Eight original normative documents recover exactly after removing glossary navigation; legal notice and two proof READMEs remain byte-identical | A new architecture or redistribution decision |
| Runtime/source protection | All 1,815 protected non-Markdown inputs unchanged; four sealed runtime artifacts and three earlier qualification archives unchanged | A newly published runtime candidate |

The checks use Markdown syntax trees and actual target/fragment lookup, not a keyword-link count.
The executable-token comparison distinguishes comments from literals/operators rather than stripping
comments with a regular expression. The two non-runnable complete classes are explicitly compile-only.
The copied programs exercise their documented behavior; this record does not relabel their runs as
new Metal, oneDNN or whole-training numerical qualification.

The actual renderer is a local QA surface, not a deployed documentation website or a claim that a
public GitHub upload occurred. Representative screenshots and real link-following observations are
retained with the evidence, including fixed recurrence versus cumulative scan, nonzero subnormals,
copy/borrow/result boundaries, and the recurrent differentiation prerequisite.

## Independent review and corrected findings

Three independent model-based review lenses covered disjoint **70**, **234** and **674** page sets,
respectively. Their final scope is the complete task baseline above, including first-pass glossary
mistakes inherited by the intermediate snapshot. Ten disjoint correction scopes supplied exact
per-occurrence actions, preserved exceptions and static source references; the coordinator integrated
them and performed the execution/rendering/preservation checks.

Corrected findings include:

- Explicit compiler constants remain structural graph inputs; only ordinary bindable inputs exclude
  them. Input borrowing extends through result closure. Eligible named length state is not the same
  role as external valid-length input data.
- Preparation constructs reusable executable recipes; Runtime later binds current run storage.
  Both fold and scatter own disjoint output ranges. Bound execution, not preparation, performs the
  native GEMM call. Runtime executes prepared tuning trials as well as final recipes.
- Engine owns representative reference-byte comparison; generic tuning orders correctness before
  timing and interprets the collaboration's outcome. `InitializedBuffer` retains a creator recipe;
  it does not prove that a representation has already been created.
- Window expressions retain an exact input Tensor, which may be a leaf, and their own producing
  occurrence. Rank-three columns describe `unfold2d`, not general-axis unfold. Zero-time residual
  history examples use retained shared carry state, not an invariant bias with no history port.
- The subnormal definition explicitly excludes zero. Fixed recurrent gate packing was restored.
  A too-wide LSTM equation was split into aligned rows. Measurement batches were restored instead
  of being mislabeled as candidate batches.
- Wrong glossary senses were removed or precisely retargeted: ordinary verbs, inspection scans,
  recurrence families, scalar execution/immediates/values, descriptor inference, feature widths,
  numeric/diagnostic payloads, resource pooling, ABI layouts, and other unrelated domains.
  Genuine cumulative operations and other explicit exceptions remain linked correctly.
- Generated glossary prose interrupting eight historical arithmetic expressions was removed.
  The preservation check caught a link-unwrapping error in the literal interval `[0,1024)`;
  the original interval and complete visible paragraph were restored. An inline-code guard checks
  the correction baseline rather than accepting a plausible-looking paragraph. A dangling
  structural-evidence sentence and a duplicate old unfold statement were also repaired.

Human-reader observations: **none**. These are source-grounded automated/model-based reviews and
actual software checks, not a usability study or a guarantee that every reader already understands
every term.

## Known evidence boundaries

- **49 existing historical `/private/tmp` evidence links are unavailable.** Their original records
  disclose that limitation. No missing experiment, report or successful result was invented or
  silently rerun to replace them.
- **191 external URL targets were not fetched.** Local file/Markdown anchors and dictionary targets
  are checked separately; an external destination is not certified by appearing in the inventory.
- The two immutable proof README files are available in the full repository copy or separately
  supplied evidence, not the source-only ZIP. No tutorial depends on reading those proof records.
- Old runtime and documentation receipts retain their original source/version boundaries. Merely
  reading, linking or clarifying them does not renew their qualification.
- Distribution remains private/internal. No public project licence or redistribution permission
  has been selected, and no public upload is authorized.

## Source delivery and reproducible evidence

The selected source list contains **2,793 unique sorted paths**: the existing selection plus this
record and the terminology packet. The source-only delivery uses version `0.1.0-terms.20261104`,
distinct from the old sealed runtime and earlier documentation versions. Its generated
`source-provenance.json` records the exact committed base and per-input hashes.

The source tasks are `sourceManifest` and `sourceSnapshot`, with explicit `releaseVersion` and
`releaseBaseRevision`. Acceptance receipts outside the source ZIP record archive CRC, exact selected
membership, live/manifest/extracted SHA-256 agreement, and exclusion of generated outputs. Keeping
those receipts outside avoids a self-referential archive hash; it does not change the selected
source inputs.

The private documentation evidence archive retains the complete page/anchor/link/math inventories,
preservation inputs, before/after comparison snapshots, JDK token tool/results, actual compile/run
commands and outputs, renderer dependency lock, DOM observations, screenshots, review reports and
per-occurrence correction receipts, including caught failures and their repairs. Temporary tooling,
extracted dependencies, browser/service state and owned worktrees are not shipped as production
implementation.
