# General documentation style

## Purpose

These rules apply to every Synaptik document. They make content understandable to a reader who knows general programming but is new to this repository.

Use these rules with one [type-specific profile](README.md). They do not change the authority of [`ARCHITECTURE.md`](../../../ARCHITECTURE.md).

## Required content and style

### Begin with purpose and a [mental model](../../glossary.md#term-mental-model)

State what the document helps the reader understand or accomplish before introducing classes, [modules](../../glossary.md#term-module), or commands. Give the reader a small organizing explanation, such as:

```text
compile -> prepare -> run
meaning    executable   invocation
and owner  state        state
```

Then add detail. A list of names without their relationships makes a newcomer infer the design.

### Use beginner-friendly language

- Assume no prior Synaptik knowledge.
- Prefer direct sentences and concrete nouns.
- Define acronyms and project-specific terms at first use.
- Explain why a step or rule matters, not only what it is called.
- Replace vague words such as “thing,” “magic,” “simple,” and “obvious” with the actual concept or condition.
- Distinguish terms that are easy to confuse, such as [tensor](../../glossary.md#tensor), [graph value](../../glossary.md#graph-value), [buffer](../../glossary.md#term-buffer), and [memory slot](../../glossary.md#memory-slot).

Do not remove necessary technical detail. Introduce it in layers: purpose, [mental model](../../glossary.md#term-mental-model), concrete example, then precise edge cases.

### Name the participants at every important call boundary

“Caller” and “owned” are roles, not sufficient explanations. Name the
[caller and callee](../../glossary.md#caller-callee-and-the-call-boundary), the API being called,
and the data crossing it. State whether an argument describes future work or contains current
values, whether the callee borrows or copies it, and who retains any cleanup responsibility.
An internal caller can be Runtime or another framework component rather than the application.

Explain composite phrases locally. For example, do not leave “caller-owned leaves” for the
reader to assemble from three dictionary entries. Identify the application or internal component,
the leaf Tensor's role, and the storage/copy boundary in that example. Link a prerequisite
definition or a focused explanation in addition to the local explanation.

### Keep terminology and the glossary aligned

Use the central [glossary](../../glossary.md) for every specialist concept, not just Synaptik-specific
names. Link the first meaningful use of each concept in every independently readable section;
readers may arrive directly at a reference heading or historical task. Do not assume knowledge
of [machine learning](../../glossary.md#term-machine-learning-ml), numerical analysis, [compiler](../../glossary.md#term-compiler) construction, native integration or [planning](../../glossary.md#term-planning)
acronyms. Expand an acronym on first use and link its definition.

Give an important prerequisite a short local explanation as well as a dictionary link. A useful
dictionary entry starts with plain-language purpose, then an example or contrast, then precise
Synaptik details. Link prerequisite concepts inside definitions; avoid circular acronym chains.
Keep one definition home and preserve existing incoming heading anchors.
An independently addressable definition must link its own prerequisites. A reader arriving at
an inline dictionary anchor has not necessarily read the preceding definition or paragraph.


Preserve [API](../../glossary.md#term-application-programming-interface-api)/source links. If the only mention already links to an API, add an adjacent definition
link or a short vocabulary note rather than replacing useful API navigation. Do not repeatedly
link the same term within one section, match an ambiguous ordinary word to the wrong specialist
meaning, or turn every method name into a separate dictionary entry.

Adjacent [lookup](../../glossary.md#term-lookup) notes are explanatory navigation, not new contract clauses. Keep an original
API link intact and label the companion with the term and “glossary”. Source markers
`glossary-navigation` delimit such notes so preservation checks can compare original [normative](../../glossary.md#term-normative)
clauses exactly after removing navigation markup.

Never insert Markdown links into executable examples, commands, mathematics, diagram source or
verbatim third-party legal text. Explain their specialist labels in surrounding prose instead.
Byte-preserved proof records have an external [reading key](../../terms.md#reading-preserved-evidence).
Historical decisions, [hashes](../../glossary.md#term-hash) and execution records retain their original assertions; added
navigation is not retrospective requalification.

Update the glossary whenever a review finds an unexplained reusable concept or a distinction that
readers need. Do not add ordinary words merely to increase a coverage count.
Prefer authored explanations over accumulated lookup parentheses. Once useful local prose and
precise links exist, remove redundant mechanical lookup notes; the number of links is not a
measure of understanding.


### Use links as navigation, not as missing explanation

- Use descriptive link text that tells the reader what the [target](../../glossary.md#term-target) contains.
- Link to the authoritative or most focused source rather than several near-duplicates.
- Explain the immediate point locally; do not write “see here” in place of an explanation.
- Prefer relative links inside the repository.
- Check file targets and heading anchors after edits.

### Make relationships visible

Use a small table, flow, diagram, or example when it makes sequence, [ownership](../../glossary.md#term-ownership-meanings), state, or comparison materially easier to understand. Each visual must teach a specific relationship and receive a sentence explaining how to read it.

Use the [example format](example-format.md) for examples with inputs, transformations, or observable results. Label [pseudocode](../../glossary.md#term-pseudocode) and planned APIs clearly.

### Write mathematics as mathematics

Use GitHub-supported LaTeX for arithmetic and calculus explanations: `$...$` inside a sentence
and `$$...$$` for a displayed derivation. Introduce every symbol, index, axis, unit and summation
range before relying on it. Show a small worked case and explain what the result means.

State the formula's domain and distinguish real-number reasoning from actual floating-point
evaluation. Do not silently apply a finite-only formula to NaN, infinity, empty domains or zero
denominators, or treat an algebraic identity as permission to reorder rounded operations.
Keep executable Java arithmetic, API identifiers, commands, hashes and recorded stdout as code.

### Put the reason beside the code

Add short comments inside examples where readers need the purpose or consequence of a step:
who supplies the input, what is only described rather than executed, what is copied or borrowed,
why resources close in that order, and whether a reported loss is before or after an update.
Do not merely repeat the method name. Keep comments truthful to the actual API.

Use the language's comment syntax; do not insert comments into strict JSON, literal output or
cryptographic evidence. Existing examples must retain their executable behavior. Comment-only
changes can be checked by token comparison; the coordinator also compiles and runs changed
copyable paths instead of assuming that prose alone proves them.

### Structure around questions the reader needs answered

Use chapters and subchapters for different questions, one idea per paragraph, and bullets for
independent conditions or steps. Introduce participants and inputs before the procedure;
interpret the result before listing edge cases. Keep a short focused page short—do not paste
the same template or glossary list into every document.

Connect the explanation to focused related topics and prerequisites. A diagram shows a genuine
relationship; a formula explains the calculation; neither substitutes for naming the participants
and explaining why the reader is seeing it.

## Avoid

- unexplained jargon, acronyms, or internal shorthand;
- circular definitions;
- long inventories without purpose or grouping;
- examples with no stated inputs, result, or interpretation;
- decorative diagrams that repeat nearby prose;
- duplicated architecture rules that can drift from the contract;
- promises about planned behavior written as if already implemented; and
- empty verbosity that restates the same fact without adding meaning.

## [Validation](../../glossary.md#term-validation)

- Confirm a newcomer can state the document's purpose after the opening section.
- Check every project term at first use and against the glossary.
- Check local links and anchors.
- Verify that examples are current or clearly marked conceptual.
- Confirm tables and diagrams have an explanatory sentence.
- [Run](../../glossary.md#term-run) `git diff --check` and check new files for trailing whitespace.

## General template

```markdown
# Descriptive title

## Purpose

What the reader will understand or accomplish and why it matters.

## Mental model

A short explanation or diagram showing the important relationships.

## Main content

Definitions, steps, contracts, or decisions required by the selected profile.

## Example or scenario

Inputs, meaningful steps, result, and interpretation when appropriate.

## Boundaries and limitations

What this document or feature deliberately does not cover.

## Related documentation

Descriptive links to authoritative and adjacent material.
```
