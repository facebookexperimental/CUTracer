# CUTracer Automated Diagnosis Service

This package owns a backend-independent, per-unit analysis session:

```text
initial campaign
  |-- compute-sanitizer sweep
  `-- CUTracer random-delay stress + correctness oracle
            |
            v
       evidence fan-in
            |
            v
       AI analyze/plan
       |           |
     final      follow-up
                   |-- reg_trace or mem_value_trace
                   `-- replay + reduce when stress saved a triggering config
                              |
                              `------> next AI turn
```

The two initial sources are peers. A clean sanitizer result never suppresses
random-delay stress. A local backend may serialize them on one GPU, while a
distributed backend may submit them concurrently on MAST; both deliver the same
typed completion events to `session.py`.

`cutracer.stress` records a unique delay config for every random-delay attempt and
retains the config whose correctness oracle reproduced the issue. The typed
`TriggeringDelayConfig` also pins the target/oracle argv, revision, kernel,
architecture, delay parameters, artifact digest, tool versions, and observed
reproduction rate. The reducer in `cutracer.reduce` is an independent follow-up
experiment that preserves the original config, minimal config, full report, and
replay failure as distinct evidence.

Reduction is not a discovery mode. The standalone `cutracer reduce` command
requires an existing delay config, and the service constructs a reduce experiment
only from a previously saved `TriggeringDelayConfig` plus its approved oracle.
Missing config is rejected before dispatch; the reducer never synthesizes a
default config or launches random-delay discovery implicitly.

The pure state machine returns effects rather than performing infrastructure
calls. `cutracer diagnose` interprets them synchronously for Local.
`cutracer.service.distributed` persists the session before dispatching MAST
experiments or a short-lived
Sandcastle reasoning turn. No MAST process calls Claude, and no Sandcastle worker
waits for a long-running GPU job.

Distributed session updates use revisioned compare-and-swap. Pending experiment
specs are part of the durable session so `resume()` can safely re-submit them;
MAST, Sandcastle, and report adapters must deduplicate their stable dispatch
keys. A Sandcastle decision includes the revision it analyzed, so a late result
cannot overwrite a newer round.

## Runtime ownership

- `cutracer.service.contracts`: pure-stdlib wire/domain contracts.
- `cutracer.service.experiments`: typed adapters over the CUTracer Python APIs.
  They never invoke a `cutracer` executable. Compute Sanitizer is the only
  vendor subprocess; stress, trace, and reduce inject the `cutracer.so` bundled
  with the current Python package.
- `cutracer.service.session`: backend-independent state transitions, budgets,
  and idempotency.
- `cutracer.service.reasoner`: structured
  `FINAL | FOLLOWUP_REQUIRED | INCONCLUSIVE` output;
  model output cannot supply arbitrary shell commands. Artifact content is read
  through a bounded loader; Local reads `file://` artifacts and Sandcastle can
  inject a loader for materialized remote artifacts.
- `cutracer.service.runner`: Local driver used by `cutracer diagnose`.
- `cutracer.service.distributed`: MAST/Sandcastle/store/report deployment ports.

An externally scheduled producer, such as a weekly TritonBench CUTracer MAST
run, should upload its artifacts and deliver the same typed experiment-result
event to the coordinator. Reading Manifold is therefore a deployment ingest
adapter, not a second execution mode in the GPU experiment APIs.

All portable artifacts use `ArtifactRef`. Local paths are materialized only in
the local adapter; distributed producers must upload them and replace the URI
before sending a completion event.
