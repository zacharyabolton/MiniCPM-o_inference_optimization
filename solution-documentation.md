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
    Average time to first byte: 
        run1: 1.339s
        run2: 1.617s
        run3: 1.420s
    Average realtime factor:
        run1: 0.825
        run2: 1.060
        run3: 0.878
```

Two of these three runs shows a nice bump, while one of them regresses back to our initial performance. This shows the variability and our perf measurements and begs for more robust benchmarking.

For now, we will just run each test thrice to obtain more robust measures.

### Flash Attention vs Scaled Dot Product

Even though flash attention is better optimized for long input lengths, it is worth trying even though we are dealing with short inputs.

The results of three runs are:

```txt
AGGREGATE PERFORMANCE METRICS:
    Average time to first byte: 
        run1: 1.873s
        run2: 1.872s
        run3: 1.955s
    Average realtime factor:
        run1: 1.066
        run2: 1.066
        run3: 1.150
```

Not good, but there's a potentially important error message in the CLI:

```
You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
/usr/local/lib/python3.11/site-packages/transformers/models/auto/image_processing_auto.py:513: FutureWarning: The image_processor_class argument is deprecated and will be removed in v4.42. Please use `slow_image_processor_class`, or `fast_image_processor_class` instead
  warnings.warn(
Loading checkpoint shards:  75%|███████▌  | 3/4 [00:03<00:01,  1.32s/it]Loading checkpoint shards: 100%|██████████| 4/4 [00:06<00:00,  1.68s/it]Loading checkpoint shards: 100%|██████████| 4/4 [00:06<00:00,  1.55s/it]
```

Reaing up on the error (https://github.com/openai/whisper/discussions/1948#discussioncomment-8074335) indicates we need to load the model onto CPU first and then move to GPU seperately.

This was done, and the following results were obtained.

```txt
AGGREGATE PERFORMANCE METRICS:
    Average time to first byte: 
        run1: 1.539s
        run2: 1.858s
        run3: 1.601s
    Average realtime factor:
        run1: 0.834
        run2: 1.076
        run3: 0.915
```

Unfortunately this is a regression, so `sdpa` it is.

### Quantization

Went down a rabbit hole trying to use a pre-quantized version of MiniCPM-0-2_6 and enabling variable quantization with `BitsAndBytes`. Because of time constraints and container image complexity, this was tossed.

## Final Benchmarks

The two optimizations kept were model compilation and model precision.

These changes were reverted and the inference was run thrice with the original implementation for comparison:

**Original metrics**

```txt
AGGREGATE PERFORMANCE METRICS:
    Average time to first byte: 
        run1: 1.639s
        run2: 1.537s
        run3: 1.593s
    Average realtime factor:
        run1: 1.099
        run2: 0.975
        run3: 1.071
```

TTFB            = ~(1.639 + 1.537 + 1.593)/3 = ~1.590
Realtime Factor = ~(1.099 + 0.975 + 1.071)/3 = ~1.048

**Optimized metrics**

```txt
AGGREGATE PERFORMANCE METRICS:
    Average time to first byte: 
        run1: 2.060s
        run2: 1.539s
        run3: 1.289s
    Average realtime factor:
        run1: 1.022
        run2: 1.000
        run3: 0.781
```

TTFB            = ~(2.060 + 1.539 + 1.289)/3 = ~1.629
Realtime Factor = ~(1.022 + 1.000 + 0.781)/3 = ~0.934

Original and optimized audio samples were saved to `audio_orig` and `audio_opt` respectively, for comparison.

TTFB: Worsened from 1.590s to 1.629s (+2.5%)
Realtime Factor: Improved from 1.048 to 0.934 (-10.9%)

Audio was spot checked for differences in quality and none were found.

_NOTE: My wife fell ill the night I recieved this take home and our usual childcare fell through. I was in and out with this task and apologize for the much longer than desirable submission time (~24h later)._