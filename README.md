<div align="center">

# insightsmith

**Forging insight from raw data.**

An agentic data consultant that runs on your own machine.



</div>

---

> ### ⚠️ 0.0.1 is a placeholder release
>
> **There is no working code in this version.** It reserves the name on PyPI and
> states the intent. Installing it gives you an empty package — no CLI, no API,
> nothing to run.
>
> Watch the repo if the idea below sounds useful. The first genuinely usable
> release is **0.1.0**, which will profile data files with no LLM involved at all.

---

## The idea

Point it at a data file. It works out what the file actually is, profiles it
properly, then uses an LLM to suggest analyses worth running, write the code,
execute that code in a sandbox, check the statistics for the usual crimes, and
hand back a report you could show someone.

It is built local-first. A local model on your own hardware is the default path,
not a degraded fallback — which means your data never has to leave the machine.

## License

Apache-2.0. See [LICENSE](LICENSE).
