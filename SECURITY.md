# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
**Security → Report a vulnerability** form on this repository (private
vulnerability reporting). Do not open a public issue for security problems.
You can expect an acknowledgement within a week.

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

## Design notes

* Credentials are read only from environment variables, never written to
  results or manifests, and redacted from error messages.
* The offline mode (default) makes no network calls.
* The Streamlit app binds to `localhost` by default and has no authentication;
  do not expose it publicly with provider credentials configured.
* CI runs a full-history secret scan (gitleaks) and a known-vulnerability audit
  of the locked dependencies (pip-audit) on every pull request and weekly.
