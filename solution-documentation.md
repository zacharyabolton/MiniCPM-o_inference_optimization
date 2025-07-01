## Profiling

First plan of attack is to add timing for each stage of the pipeline in order to identify bottlenecks.

`perf_counter()` is already being leveraged for realtime factor and time to first byte calcs. We can hook into this for additional measurements.

- Model load
- TTS initialization
- Prefill
- Generation
- Audio processing

### Observations

See `logs/profiling_1.txt`.

- Model load: 35.831s slow, but one-time
- TTS initialization: 0.294s not a bottleneck
- Prefill: 0.027-0.030s not a bottleneck
- AVG Generation: 8.167s
    - First response time consistently ~1.5s
- Audo processing: ~0.000 not a bottleneck

_Key Takaway_

**Generation time is the primary bottleneck**