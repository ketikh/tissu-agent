# 001: Tech Stack Selection

## Date
2026-03-26

## Status
accepted

## Context
Building a local-first AI agent system for business support, sales, and marketing.

## Decision
- Python 3.11+ for backend
- FastAPI for HTTP API
- SQLite for local data storage
- Anthropic Claude API for LLM capabilities
- N8N for workflow orchestration

## Reasoning
- Python: best ecosystem for AI/ML, FastAPI is modern and fast
- SQLite: zero-config, local-first, good enough for single-instance deployment
- Claude API: strong reasoning for agent tasks
- N8N: visual workflow builder, easy for non-technical users

## Consequences
- Single-instance deployment (SQLite limitation)
- Requires ANTHROPIC_API_KEY for LLM features
- N8N runs separately on port 5678
