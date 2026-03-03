# CI and Quality Gates

This repository uses GitHub Actions for pull-request quality gates.

## What runs on pull requests

- `Lint`: `make -f .github/Makefile lint`
- `Unit Tests (py3.10)`
- `Unit Tests (py3.11)`
- `Integration Tests`

If any of these checks fail, the PR should be blocked from merge using branch protection.

## What runs on pushes to `main`

- `Lint`
- `Unit Tests (py3.10, py3.11)`
- `Integration Tests`

## Security and maintenance automation

- Weekly dependency audit workflow: `.github/workflows/security.yml`

## Local commands (match CI)

```bash
pip install -r .github/requirements-dev.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch

make -f .github/Makefile lint
make -f .github/Makefile test-unit
```

## Enable PR blocking (GitHub settings)

In `Settings -> Branches -> Add branch protection rule` for `main`:

1. Enable `Require a pull request before merging`.
2. Enable `Require status checks to pass before merging`.
3. Add required checks:
   - `Lint`
   - `Unit Tests (py3.10)`
   - `Unit Tests (py3.11)`
   - `Integration Tests`
4. Enable `Require branches to be up to date before merging`.
5. (Recommended) Enable `Require approvals` with at least 1 reviewer.

## Files added for this setup

- `.github/workflows/ci.yml`
- `.github/workflows/security.yml`
- `.github/CODEOWNERS`
- `.github/pull_request_template.md`
- `.github/pre-commit-config.yaml`
- `.github/requirements-dev.txt`
- `.github/Makefile`
