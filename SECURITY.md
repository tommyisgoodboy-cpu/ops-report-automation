# Security policy

This is a portfolio project that runs on synthetic data. It holds no real customer, employer or government data.

## Reporting a vulnerability

Please **report it privately**, not in a public issue: use the **Security** tab of this repository and choose
**Report a vulnerability** (GitHub private vulnerability reporting). Include what you found, how to reproduce it and
what you think the impact is. I will answer on a best-effort basis.

## In scope

- The code in this repository.
- The live demo's behaviour: injection, cross-site problems, ways around the rate limits or the daily caps,
  anything that exposes a credential or a personal detail.

## Out of scope

- Load or denial-of-service testing against the live demo. It runs on a small, capped, pay-per-use service.
- Findings that need a stolen credential or physical access to a machine.

## What the project already does

See "Security notes" and "Remaining risks" in the [README](README.md): keyless authentication with a dedicated
service account, input validation, rate limiting, daily caps on paid models, security headers, pinned dependencies,
weekly Dependabot updates and a `pip-audit` step in CI.
