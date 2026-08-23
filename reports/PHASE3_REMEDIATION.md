# Phase 3 Remediation — GPT-5.6 Terra

This report records corrections made after the immutable
`phase-3-review-candidate` (`20ef252cda381382b979ef4e98693184eef7441d`).
The candidate tag was not moved.

| ID | Severity | Symptom and root cause | Remediation | Regression evidence |
|---|---|---|---|---|
| T-01 | HIGH | `factorial(5000)` successfully computed but Python's integer-to-string safety limit raised `ValueError` during result formatting; the runtime converted this ordinary bounded-input failure into `INTERNAL_ERROR`. | Added an 8,192-byte canonical-result limit; integer bit length is checked before formatting and the dispatch pipeline rejects oversized serialized results as `RESOURCE_LIMIT`. | `test_factorial_at_configured_compute_limit_is_bounded_by_result_limit`; direct runtime attack returns `RESOURCE_LIMIT`. |
| T-02 | MEDIUM | Conversion and finance operations inherited the mutable process-global Decimal context, so a caller could change output precision. | Both public Decimal computation paths now use an explicit precision-28, `ROUND_HALF_UP` local context; Decimal failures are translated to stable protocol errors. | `test_conversion_is_independent_of_process_global_decimal_context`; `test_finance_is_independent_of_process_global_decimal_context`. |
| T-03 | MEDIUM | The advertised 512-character expression limit was unreachable because generic JSON string validation rejected every string over 256 characters first. | Harmonized the generic bound at 512 characters and documented the actual enforcement order. | Existing expression-bound and parser-limit tests, plus configuration/hash validation. |

All changes remain within the unapproved `juniper-tool-protocol-v1` 1.0.0
contract. No Phase 1 architecture artifact or Phase 2 tokenizer artifact was
modified.
