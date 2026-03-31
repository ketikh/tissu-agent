# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it privately.

**Do NOT create a public GitHub issue for security vulnerabilities.**

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |

## Security Practices

- All secrets stored in environment variables
- Input validation on all API endpoints
- Parameterized database queries only
- No hardcoded credentials in source code
- Dependencies audited regularly with `pip audit`
