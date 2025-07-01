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

### Model Precision

We're currently using bfloat13 but we can optimize this more using TF32 since we're on an A10G GPU: https://docs.pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices 

The results of doing so are:

```txt
AGGREGATE PERFORMANCE METRICS:
   Average time to first byte: 1.339s
   Average realtime factor: 0.825
   Average generation time: 10.069s
   Average audio processing time: 0.000s
   TTFB std dev: 0.010s
   Realtime factor std dev: 0.073
```

```txt
AGGREGATE PERFORMANCE METRICS:
   Average time to first byte: 1.617s
   Average realtime factor: 1.060
   Average generation time: 8.089s
   Average audio processing time: 0.000s
   TTFB std dev: 0.038s
   Realtime factor std dev: 0.034
```

```txt
AGGREGATE PERFORMANCE METRICS:
   Average time to first byte: 1.420s
   Average realtime factor: 0.878
   Average generation time: 6.093s
   Average audio processing time: 0.000s
   TTFB std dev: 0.069s
   Realtime factor std dev: 0.049
```

Two of these three runs shows a nice bump, while one of them regresses back to our initial performance. This shows the variability and our perf measurements and begs for more robust benchmarking.

For now, we will just run each test thrice to obtain more robust measures.