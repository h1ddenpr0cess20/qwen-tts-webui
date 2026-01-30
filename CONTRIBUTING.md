# Contributing

Thanks for your interest. This is primarily a solo learning project. That said, small, focused contributions that fix real bugs are welcome.

## What’s Welcome
- Bug fixes with clear reproduction steps.
- Small documentation tweaks that improve accuracy or clarity.

## What’s Not a Fit
- New features or large refactors (unless discussed and agreed in advance).
- Off-topic discussions (political/ideological/social). Keep it technical.

## How to Contribute
1. Fork the repo and create a branch for your fix.
2. Reproduce the issue locally and confirm the root cause.
3. Make a minimal change that fixes the problem without unrelated edits.
4. Manually smoke test:
   - API: `uvicorn app.main:app --reload --port 8000`
   - UI: open `http://localhost:8000` (served by the app) or `frontend/index.html` for layout checks.
   - Docker (optional): `docker compose up --build` for parity with docs.
5. Open a pull request describing:
   - The bug and steps to reproduce
   - The minimal fix you applied
   - Any notes on limitations or follow-ups

## Coding Style
- Python: type hints preferred; run existing linters/formatters if configured; avoid adding new dependencies.
- JS/HTML/CSS: semantic class names; keep vanilla JS modules small; avoid new dependencies unless discussed.
- Don’t commit secrets. Keep configs in env vars; follow `docs/configuration.md`.

## Tests
- No automated tests are configured; please provide clear repro steps and a manual browser/API smoke test.

## Maintainer Notes
- Maintainers may close off-topic or out-of-scope PRs/issues to keep focus on learning and code quality.

If you found a bug and can fix it—great. Open a PR. Otherwise, please file concise, actionable issues only when they relate to concrete technical problems.
