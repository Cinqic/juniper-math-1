# Secrets Policy

- No credentials, API keys, tokens, SSH keys, or passwords are ever
  committed to this repository.
- `.gitignore` excludes `.env` and common key/cert extensions.
- No `.env.example` file exists in Phase 0 because no environment variables
  are currently required by any Phase 0 command. If a later phase
  introduces one (e.g. an API key for an external tool), add
  `.env.example` with placeholder values only — never real secrets — at
  that time.
- Before every commit, review `git status` / `git diff --staged` for
  anything that looks like a credential, even in an innocuous-looking
  filename.
- If a secret is ever discovered in Git history, report it immediately
  (it must be treated as compromised and rotated) rather than merely
  deleting the current copy — history rewriting is a separate, deliberate
  remediation, not a quiet fix.
