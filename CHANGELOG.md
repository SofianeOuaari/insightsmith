# Changelog

All notable changes to this project are documented here. This file is generated
by [git-cliff](https://github.com/orhun/git-cliff) from the commit history.

## [0.4.0] - 2026-08-09

### Build and packaging

- Bump to 0.4.0 and document the the dataset card

### Features

- **profiling**: Add the dataset card and best-effort PII masking.
- **cli**: Add `ismith look --ideas` and --card

## [0.3.0] - 2026-08-03

### Build and packaging

- Add httpx and tomli, bump to 0.3.0, document the provider layer

### Features

- **llm**: Add the provider layer, config and capability-aware router
- **cli**: Add `ismith models`

### Housekeeping

- Ignore the local example dataset

### Testing

- Fail any test that opens a real HTTP connection

### Chorse

- **release**: V0.3.0

## [0.2.0] - 2026-07-29

### Build and packaging

- Add psutil, bump to 0.2.0, document ismith doctor

### Features

- **hardware**: Probe the machine and size models against it
- **cli**: Add `ismith doctor`

## [0.1.1] - 2026-07-29

### Bug fixes

- **profiling**: Infer date formats after loading so a bad date cannot fail the read

### Build and packaging

- Bump version to 0.1.1

### Documentation

- Note that date formats are inferred and may be ambiguous

## [0.1.0] - 2026-07-27

### Bug fixes

- **sniff**: Read sparse non-ASCII as cp1252 and find headers after comments

### Build and packaging

- Add packaging, tooling, CI and release workflows
- Configure git-cliff and package the changelog

### Documentation

- Add initial README
- Add design specification
- Add CLAUDE.md engineering rules
- Rewrite the README for the 0.1.0 foundation

### Features

- **sniff**: Detect source format via extension, magic bytes and dialect probe
- **io**: Load csv, excel, parquet, arrow and json behind one interface
- **profiling**: Add schema, statistics and quality checks
- **cli**: Add `ismith look` with rich tables and --json

### Housekeeping

- Stop tracking docs/ in git
- Stop tracking CLAUDE.md in git
- Ignore local-only CLAUDE.md and docs/

### Testing

- Close the sqlite fixture connection explicitly

