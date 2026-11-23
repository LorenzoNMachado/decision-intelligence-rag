# Security review

This review records the repository state audited for portfolio hardening. It
does not certify the prototype as secure. The frozen experiment manifests,
protocols, evaluation sets and historical result files were not changed.

## Findings and current controls

| Area | What the code currently does | Limit |
|---|---|---|
| Prompt injection | The optional answer prompt tells the model to use the supplied decision records only. | Retrieved text is not explicitly marked as untrusted data with no instruction authority. No provider-backed adversarial evaluation has run; configuration L remains unexecuted. Treat this as an open research risk. |
| Evidence poisoning | The evidence gate and sufficiency checks can abstain; the v3 audit records adversarial cases and known failures. | The audit found unsupported answers for rejected, withdrawn, contradictory and wrong-attribute facts. The results on generated holdout questions do not establish robustness to poisoned records. |
| Citation integrity | Claims must cite admitted record IDs; quote enforcement checks that a sufficiently long quote occurs in the cited record body. | A literal quote proves traceability, not entailment. The optional LLM claim verifier is experimental and has no reported provider run. |
| Evidence sufficiency | Offline rules can return `ABSTAIN` when evidence is not admitted or does not match the requested fact. | The rule set has documented paraphrase, scope, contradiction and subject-resolution failures. Preserve `ANSWER`, `ABSTAIN` and `ERROR` as separate outcomes. |
| Data isolation | The demo uses a bundled synthetic corpus and runs locally by default. | There is no authentication, tenant model or cross-user isolation. Do not expose the app as a shared service or use sensitive organizational records. |
| Input and paths | JSON corpus loading and corpus integrity checks validate repository-format data. CLI paths are supplied by the local operator. | The project is not a sandbox for hostile files or a multi-tenant upload service. Only load files from trusted local sources. |
| Provider security | Provider keys come from environment variables; error text is sanitized; tests use mocked transports. Offline mode makes no provider calls. | With an LLM provider enabled, questions and admitted evidence leave the local process. Provider retention, access controls and logging are outside this repository's control. |
| Reproducibility | CI checks the frozen manifests, corpus generation, and historical v1/v2 records. | These integrity checks establish that committed inputs match their manifests; they do not establish scientific validity or security. |

## Adversarial test boundary

Existing tests cover malformed provider responses, secret redaction, invalid
citations, quote matching, answer states and evidence sufficiency. A separate
test records that a valid source ID plus a literal quote can still leave claim
support unchecked; this is deliberately outside the frozen experiment
protocols. Prompt-injection resistance cannot be demonstrated with the
scripted provider used by the suite, and adding a provider prompt or changing
the hashed runtime source would alter the frozen inputs. That work belongs in
a new, separately registered protocol with independent adversarial cases.

## Current automated checks

The security workflow scans full Git history with redacted Gitleaks output,
audits the locked Python dependencies with pip-audit and runs Bandit. CodeQL
runs on pull requests without uploading results or receiving write permission;
on pushes to `main` and on a schedule, only its analysis job receives
`security-events: write` to upload results. These scanners have different
coverage and can miss vulnerabilities.

## Static-analysis scope

CI gates Bandit findings at medium severity or higher. The current code has
low-severity findings for Git subprocess calls in historical protocol and
manifest verification. Those calls use fixed Git argument lists, do not invoke
a shell, capture output, and set timeouts. The findings are retained as a
documented review item rather than changing source files whose hashes are part
of the frozen research record.
