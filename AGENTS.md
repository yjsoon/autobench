# AGENTS.md

Read CLAUDE.md

## Benchmark Monitoring

- When checking long-running engine startup or benchmark jobs, prefer **narrow status checks** over
  dumping whole logs into the context.
- First check the smallest signal that answers the question:
  - endpoint readiness such as `curl -fsS http://localhost:<port>/health`
  - container/process liveness via `docker ps`, `docker inspect`, or `docker stats --no-stream`
  - targeted log grep such as `docker logs ... | rg "Application startup complete|ERROR|Traceback|Killed"`
- Only read broader log output when a narrow check shows an actual failure or when the expected marker
  cannot be found with targeted grep.
- For startup progress, grep for specific readiness or failure markers instead of ingesting the whole
  container log.
