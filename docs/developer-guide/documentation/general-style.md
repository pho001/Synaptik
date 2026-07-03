# General documentation style

## Purpose

These rules apply to every Synaptik document. They make content understandable to a reader who knows general programming but is new to this repository.

Use these rules with one [type-specific profile](README.md). They do not change the authority of [`ARCHITECTURE.md`](../../../ARCHITECTURE.md).

## Required content and style

### Begin with purpose and a mental model

State what the document helps the reader understand or accomplish before introducing classes, modules, or commands. Give the reader a small organizing model, such as:

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
- Distinguish terms that are easy to confuse, such as tensor, graph value, buffer, and memory slot.

Do not remove necessary technical detail. Introduce it in layers: purpose, mental model, concrete example, then precise edge cases.

### Keep terminology and the glossary aligned

Use the central [glossary](../../glossary.md) for Synaptik domain terms. At first use, provide a short local definition when the term is important to the current explanation, even if you also link to the glossary.

Update the glossary when a change introduces a new reusable domain term, changes the meaning or boundary of an existing term, or reveals a common distinction that readers need. Do not add ordinary programming words merely to increase glossary coverage.

### Use links as navigation, not as missing explanation

- Use descriptive link text that tells the reader what the target contains.
- Link to the authoritative or most focused source rather than several near-duplicates.
- Explain the immediate point locally; do not write “see here” in place of an explanation.
- Prefer relative links inside the repository.
- Check file targets and heading anchors after edits.

### Make relationships visible

Use a small table, flow, diagram, or example when it makes sequence, ownership, state, or comparison materially easier to understand. Each visual must teach a specific relationship and receive a sentence explaining how to read it.

Use the [example format](example-format.md) for examples with inputs, transformations, or observable results. Label pseudocode and planned APIs clearly.

## Avoid

- unexplained jargon, acronyms, or internal shorthand;
- circular definitions;
- long inventories without purpose or grouping;
- examples with no stated inputs, result, or interpretation;
- decorative diagrams that repeat nearby prose;
- duplicated architecture rules that can drift from the contract;
- promises about planned behavior written as if already implemented; and
- empty verbosity that restates the same fact without adding meaning.

## Validation

- Confirm a newcomer can state the document's purpose after the opening section.
- Check every project term at first use and against the glossary.
- Check local links and anchors.
- Verify that examples are current or clearly marked conceptual.
- Confirm tables and diagrams have an explanatory sentence.
- Run `git diff --check` and check new files for trailing whitespace.

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
