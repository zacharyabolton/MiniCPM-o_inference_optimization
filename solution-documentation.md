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

```txt
AGGREGATE PERFORMANCE METRICS:
   Average time to first byte: 1.598s
   Average realtime factor: 1.062
   Average generation time: 8.167s
   Average audio processing time: 0.000s
   TTFB std dev: 0.032s
   Realtime factor std dev: 0.055
```

_Key Takaway_

**Generation time is the primary bottleneck**

## Optimizations

### Compilation

First and easiest-win optimization is torch compilation.

We just need to call torch.compile on the model and tts after loading (with an optional mode parameter). This compiles torch code for CUDA using JIT-compilation for performance.

We will try all three `mode`s outlined at: https://pytorch.org/get-started/pytorch-2-x/#user-experience

- "reduce-overhead"
- "max-autotune"
- unset

**"reduce-overhead"**

```txt
AGGREGATE PERFORMANCE METRICS:
   Average time to first byte: 1.475s
   Average realtime factor: 0.930
   Average generation time: 9.341s
   Average audio processing time: 0.000s
   TTFB std dev: 0.032s
   Realtime factor std dev: 0.062
```

**"max-autotune"**

```txt
AGGREGATE PERFORMANCE METRICS:
   Average time to first byte: 2.223s
   Average realtime factor: 1.271
   Average generation time: 7.633s
   Average audio processing time: 0.000s
   TTFB std dev: 1.410s
   Realtime factor std dev: 0.518
```

**unset**

```txt
AGGREGATE PERFORMANCE METRICS:
   Average time to first byte: 1.570s
   Average realtime factor: 1.052
   Average generation time: 6.252s
   Average audio processing time: 0.000s
   TTFB std dev: 0.060s
   Realtime factor std dev: 0.035
```

While the differences in many of the metrics are all close enough to be noise [^1], we will go with "reduce-overhead" given it's lower TTFB and realtime factor.

[^1]: For production, we would run these exps many times to obtain statistically meaningful metrics.
