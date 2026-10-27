"""Stage 2+: Allowlisted, validated task operations.

Mutations will run in SQLite transactions; multi-action requests will
be validated and resolved before a batch is committed.
"""
