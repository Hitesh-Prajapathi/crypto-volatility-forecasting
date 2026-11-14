# 0001. Record architecture decisions

- Status: Accepted
- Date: 2026-11-13

## Context

This project is being refactored from a research prototype into a production-shaped system. Many choices (storage format, orchestration, deployment model) have non-obvious trade-offs, and the reasons behind them are lost quickly if they only live in commit messages or in someone's head.

## Decision

We will record each significant architectural decision as a short Markdown ADR in `docs/adr/`, following [the template](0000-template.md) (Status / Context / Decision / Consequences / Alternatives considered). ADRs are numbered sequentially, never deleted, and superseded by a new ADR rather than rewritten. The index lives in [README.md](README.md).

## Consequences

- Decisions are reviewable in pull requests, like code.
- Newcomers (and interviewers) can see why, not just what.
- A small amount of writing overhead per decision; ADRs are kept to about one page.

## Alternatives considered

- Wiki or Notion pages: drift away from the code and are not versioned with it.
- Only commit messages: hard to discover, no structure.
