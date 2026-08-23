# Phase 5 Remediation

| Finding | Correction | Verification |
| --- | --- | --- |
| Valid YAML could silently request unsupported or unsafe runtime behavior. | Added semantic validation for sizes, scheduler/resume bounds, optimizer, device, precision, numeric ranges, and prompts. | Full test suite, Ruff, and Mypy pass. |
| Full training resume could accept incompatible runtime configuration. | Compare checkpoint and requested full-resume configuration before restoration. | Resume gate passes exactly. |
| A non-finite clipping norm was not rejected. | Check clipping result before optimizer update. | Trainer tests and smoke run pass. |
| Resume history metric omitted pre-interruption steps. | Compare the combined stage-1/stage-2 logical history. | 200-step comparison difference is 0.0. |
| Tool evaluation default was only 20 cases. | CLI default now evaluates the complete frozen suite. | 185 cases executed. |
| Output rotation made a clean source appear dirty. | Capture source identity before mutable output operations. | Acceptance log records `source_tree_state: clean`. |
