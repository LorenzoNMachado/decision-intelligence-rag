# Security policy

## Supported versions

Only the `main` branch is maintained. There are no released versions or
release branches with separate security support. CI currently validates
Python 3.12 and 3.13.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
**Security → Advisories → Report a vulnerability** form for this repository.
Do not open a public issue or publish exploit details before reporting
privately. Include the affected component, impact and minimal reproduction
steps; do not attach real organisational records, credentials or provider
responses containing sensitive data.

This is a research project maintained on a best-effort basis. No response or
fix timeline is promised. Public issues are appropriate for non-security bugs.

## Scope

This is a research prototype, not a production service. Relevant reports
include, for example:

* credentials or personal data in the repository or its history;
* ways in which provider error messages, results or manifests could expose an
  API key;
* the spending limit allowing paid calls beyond its documented guarantees
  (see "What the spending limit does and does not guarantee" in the README);
* unsafe handling of provider responses or of untrusted text in the Streamlit
  app.

The optional LLM path has not been executed in the reported experiments. The
current answer prompt asks the model to use only decision records, but does
not explicitly establish a separate trust boundary for instructions embedded
in retrieved text. Citation and quote validation establishes traceability,
not semantic entailment. Do not use this prototype with hostile or sensitive
corpora or expose the unauthenticated app to a network.

## Design notes

* Credentials are read only from environment variables, never written to
  results or manifests, and redacted from error messages.
* The offline mode (default) makes no network calls.
* The Streamlit app binds to `localhost` by default and has no authentication;
  do not expose it publicly with provider credentials configured.
* The bundled corpus is synthetic. There is no multi-user or tenant isolation;
  this repository does not provide a secure boundary for user-supplied corpora.
  When a provider is configured, questions and admitted evidence are sent to
  that endpoint under its data-retention terms.
* CI runs a full-history secret scan (gitleaks) and a known-vulnerability audit
  of locked dependencies (pip-audit), Bandit and CodeQL. These checks are
  limited to the code and dependency versions they inspect and do not establish
  that the prototype is secure.
