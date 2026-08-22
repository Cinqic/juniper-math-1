# Git / Repository Policy

## Canonical state

GitHub (`https://github.com/Cinqic/juniper-math-1`) is the canonical state
of Juniper Math 1. Local storage — including this machine — is disposable.
If it matters, it must be committed and pushed.

## Never

- Force-push.
- Rewrite published history.
- Delete remote branches.
- Reset away another contributor's work.
- Commit secrets: API keys, credentials, tokens, SSH keys, passwords.
- Commit large disposable binaries (datasets, checkpoints) directly into Git
  history — see [`CHECKPOINT_POLICY.md`](CHECKPOINT_POLICY.md).

## Branching

`main` is the default branch. Phase work is developed and reviewed before
merging to `main`; `main` should always be in a state a fresh clone can
build, test, and validate successfully.

## Tags

- `phase-0-review-candidate` — a non-final tag marking a commit ready for
  Opus 5 independent review. Does **not** imply approval.
- `phase-0-foundation` — reserved for the final commit after Opus 5 review
  and Cinqic's final human inspection both approve. Do not create this tag
  before both approvals exist.

## Commit messages

Lowercase, imperative, phase-prefixed where relevant, e.g.:

```
phase 0: establish Juniper Math 1 foundation and recovery infrastructure
```

## Before committing

- `git status` — confirm nothing unintended (caches, venvs, secrets, scratch
  files) is staged.
- `git diff --staged` — review the actual content, not just filenames.
