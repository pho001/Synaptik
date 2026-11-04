# Reader-accessible terminology across the documentation

## Authorization and status

Status: Active. The user explicitly requested a second pass over **all** documentation: assume no
specialist vocabulary, link each technical concept to a readable central definition, and add
useful diagrams. Classification A: presentation only; no numerical, [API](../glossary.md#term-application-programming-interface-api), [ownership](../glossary.md#term-ownership-meanings), [persistence](../glossary.md#term-persistence),
[backend capability](../glossary.md#backend-capability), architecture authority or public licensing change.

Common base: `2906615da17204b8166a827a82b352bd4f13e5f6`.
The coordinator verified the completed redesign prerequisite, exact source inventory and disjoint
write/semantic scopes below before launch. All seven original editorial frontiers were [Ready](../glossary.md#term-ready) and
authorized at launch; their handoffs are now integrated. The initial reviewed selection contains
975 Markdown documents. The live census additionally found two byte-preserved proof README files;
they remain [immutable](../glossary.md#term-immutable) and have a glossary reading key outside the evidence archive.

## Shared contract

- Keep one central dictionary at `docs/glossary.md`; retain every existing heading/explicit anchor.
  Improve opaque entries with a plain-language opening, an example or useful distinction, and
  links to prerequisites. Preserve precise details and implementation-status distinctions.<span data-glossary-navigation="true"> ([opaque glossary](../glossary.md#term-opaque))</span>
- Discover and explain missing ML, tensor, numerical, compiler, memory/lifetime, backend, tooling
  and planning terms. Expand acronyms; do not replace one unexplained acronym with another.<span data-glossary-navigation="true"> ([ML glossary](../glossary.md#term-machine-learning-ml); [tensor glossary](../glossary.md#tensor); [compiler glossary](../glossary.md#term-compiler); [lifetime glossary](../glossary.md#term-lifetime); [backend glossary](../glossary.md#backend); [planning glossary](../glossary.md#term-planning))</span>
- Link the first meaningful use of every recognized technical concept in each independently
  readable section, including short reference pages and historical records. Avoid repetitive
  links in one section and avoid ambiguous ordinary-word matches. Existing API/source links stay.<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span>
- Preserve incoming anchors, program behavior, commands, recorded outputs, provenance hashes and
  historical assertions. The later user request authorizes explanatory code comments, LaTeX
  mathematics and clearer structure in current documentation; it does not authorize changing
  executable tokens or rewriting archived proof records.
- Diagrams explain a genuine relationship, lifecycle, distinction or geometry; include accessible
  title/description and a sentence explaining how to read them. Reuse a focused existing visual
  instead of duplicating the same diagram. The coordinator renders actual changed diagrams.<span data-glossary-navigation="true"> ([lifecycle glossary](../glossary.md#lifecycle))</span>
- Normative documents receive navigation-only links and marked explanatory lookup notes.
  Original clauses and code stay exact after removing link wrappers and those navigation notes.
  Historical records receive navigation, not retrospective status/implementation rewrites.<span data-glossary-navigation="true"> ([Normative glossary](../glossary.md#term-normative); [lookup glossary](../glossary.md#term-lookup))</span>
- Editors skip build/tests/lint/formatters. The coordinator performs one integrated Markdown,
  anchor, terminology, source-preservation and visual check. No unchanged Java suite is rerun.
- Integration owner: coordinator for glossary proposal reconciliation, AST-positioned link
  migration across all documents, shared navigation/style, source selection, final evidence.<span data-glossary-navigation="true"> ([owner glossary](../glossary.md#term-owner); [AST glossary](../glossary.md#term-abstract-syntax-tree-ast))</span>
- Preserve third-party legal text and the two immutable proof README files byte-for-byte.
  Review their terminology through <span data-glossary-navigation="true"> ([immutable glossary](../glossary.md#term-immutable))</span>`docs/terms.md#reading-preserved-evidence`, not by editing
  archived proofs or renewing their execution claims.

## [Ready](../glossary.md#term-ready) editorial briefs

All rows: Depends on completed redesign; Conflicts with other rows: none within exclusive [scope](../glossary.md#term-scope);
Parallel group `DOC-TERMS-2026-11-04`; Integration order Any, then coordinator linking; Integration
[validation](../glossary.md#term-validation) is the common gate above. Each writer uses its assigned isolated worktree. Shared
contract above plus its exact row and user request is the compact execution packet.

| Brief | Exclusive write scope | Complete change and [handoff](../glossary.md#term-handoff) |
|---|---|---|
| TERMS01 glossary | `docs/glossary.md` | Review all 306 existing entries. Make dense openings readable, retain details/anchors, add needed foundational definitions and prerequisite links. Reconcile other slices' proposals. Return an explicit [vocabulary-alias](../glossary.md#term-vocabulary-alias)→[canonical](../glossary.md#term-canonical-form)-heading catalogue, reviewed entry accounting and diagrams added. |
| TERMS02 learning | `docs/tutorials/**`, `docs/how-to/**`, `docs/user-guide/**`, `docs/examples/README.md` | Read every assigned page. Explain prerequisite concepts locally, not just by acronym; improve contextual distinctions and add genuinely missing diagrams. Propose missing dictionary entries with examples/source evidence. Do not change runnable code or stdout. |
| TERMS03 [tensor](../glossary.md#tensor) reference | `docs/reference/tensor/**` | Review every tensor topic for [opaque](../glossary.md#term-opaque) numerical/[shape](../glossary.md#shape)/indexing vocabulary. Add useful short introductions/distinctions/diagrams without copying [normative](../glossary.md#term-normative) detail; propose missing dictionary terms with context. |
| TERMS04 execution reference | `docs/reference/**` except `tensor/**` | Review [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/training/data/text/[checkpoint](../glossary.md#term-checkpoint)/[compiler](../glossary.md#term-compiler)/[prepare](../glossary.md#prepare)/[runtime](../glossary.md#term-runtime)/[Engine](../glossary.md#term-engine) topics. Make boundaries and vocabulary understandable; add needed diagrams and dictionary proposals without changing callable contracts or code. |
| TERMS05 architecture | `docs/explanation/**`, non-contract `docs/architecture/*.md` | Review package atlas and architectural explanations. Explain [dataflow](../glossary.md#term-dataflow), memory, compiler and native terminology in context; add only needed diagrams. Do not edit the six normative contracts. |
| TERMS06 [backends](../glossary.md#backend) | `docs/backend-guide/**` | Review all backend guides, including numerical and native sections. Explain jargon/abbreviations and needed route/[precision](../glossary.md#term-precision)/[lifetime](../glossary.md#term-lifetime) relationships without broadening [qualification](../glossary.md#term-qualification). Propose terms with actual source context. |
| TERMS07 contributor | `docs/developer-guide/**` except shared `documentation/general-style.md`, `docs/design/**` | Review contributor guides/profiles and all decision/design records. Add contextual explanation to current guides, navigation-only glossary links to historical decisions, and needed diagrams. Preserve historical decisions/status. |

Read-only historical discovery slices cover all [planning](../glossary.md#term-planning)/backend, planning/[module](../glossary.md#term-module) and
planning/extension/tool/history records. They propose missing vocabulary and report ambiguity;
the coordinator applies navigation uniformly without rewriting historical meanings. Repository
root/module README files, shared style, normative links, `docs/terms.md` and planning navigation
are coordinator-owned.

## Completion evidence

The final record will state the exact complete-document census, dictionary entries and [vocabulary aliases](../glossary.md#term-vocabulary-alias),
actual inline-link coverage, preserved original anchors/code/math, full local-link results,
rendered diagrams and independent reader-oriented review limits. Automated coverage checks do
not establish that a real human understood every definition; human-reader observations remain
separate. The [sealed](../glossary.md#term-sealed) audit and previous documentation source/evidence archives remain unchanged.

## Reader-comprehension continuation authorized by the user

The user reported unexplained or superficial passages, specifically “caller-owned leaves,” and
requested local participant/call explanations, LaTeX arithmetic/calculus, comments inside code
examples, and clearer chapters, subchapters, paragraphs and bullets. Dictionary links alone do
not satisfy this continuation. Direct related-topic links, the GitHub-facing root README and an
independent final review remain required, in that order after the teaching edits.

Classification A remains: explain existing behavior, not add capability or policy. Current
guides can be rewritten for understanding; original normative clauses, legal text and immutable
proofs remain protected. Historical records receive clearly separated reading context and
navigation, not retroactive claims. Original command/output/evidence bytes remain exact.

Each current explanation identifies the actual caller, callee and API boundary where relevant,
distinguishes a Tensor description from stored values, and states borrow/copy/ownership and
lifetime consequences. Comment-only code edits preserve executable tokens. Mathematical
explanations use LaTeX with defined symbols, domains, dimensions and a worked interpretation.
Authors remove confusing or duplicate lookup boilerplate and connect focused related pages.

The coordinator owns shared style, First terms, root/module READMEs, the final source selection,
qualification evidence and integration. The following disjoint writer scopes become Ready after
the shared caller/glossary/style prerequisite is committed; the coordinator records their exact
common revision and worktrees at launch. Every writer skips builds, tests, lint and formatters.

| Brief | Exclusive write scope | Reader-comprehension acceptance |
|---|---|---|
| READ01 dictionary | `docs/glossary.md` | Review definitions as independently entered explanations; explain prerequisites locally, preserve precise details and old anchors, use LaTeX for taught mathematics, and remove misleading sense links. Own all shared dictionary edits during the continuation. |
| READ02 learning | Original learning frontier | Explain the application/API/data roles through the connected project, annotate copyable examples with reasons, use LaTeX for the learned objective/update, structure the narrative and connect focused references. |
| READ03 tensor | Original Tensor-reference frontier | Teach geometry, indexing and numerical meaning before constraints; explain symbols and domains, annotate examples, add focused subheadings and related operations without changing numerical contracts. |
| READ04 execution | Original non-Tensor-reference frontier | Explain NN/state, Compiler, Prepare, Runtime, Engine, Data/Text, Training and Checkpoint call boundaries; annotate examples and distinguish descriptions, values, ownership and recovery. |
| READ05 systems | Original explanation/architecture and backend frontiers, excluding normative contracts | Explain actual component collaborations and resources, remove opaque diagram labels, teach numerical/cost formulas, and connect current focused implementation explanations. |
| READ06 contributor | Original contributor/design frontier, excluding shared general style | Explain workflows and tool roles, comment instructional commands/code appropriately, structure current guides and connect authoritative sources; preserve decision history. |
| READ07 backend history | Original backend-planning frontier | Review the full frontier for missing context and relevant current-topic navigation. Keep original statuses, command/output/evidence and historical assertions; add clearly separated reading keys where needed. |
| READ08 other history | Original module/extension/tool-planning frontier, excluding this packet and its final validation record | Review the full frontier and connect task/component concepts to focused explanations; add separated role/context keys without rewriting old decisions or evidence. |

The coordinator verifies the complete live Markdown/link/anchor inventory, original authority and
source-input protections, comment-only executable equivalence, representative compiled/running
copyable examples, actual LaTeX/diagram rendering and reader navigation. Independent final
reviews assess comprehension, source fidelity and mathematical correctness; their bounds and
unresolved external historical evidence are stated explicitly rather than replaced by link counts.
