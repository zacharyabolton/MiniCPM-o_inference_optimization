On 25-07-12 I took a second look at this project.

I did some basic cleanup:

- Removed a call to `super().__init__()` in the `MiniCPMo` class, as it doesn't inherit from anything and was unecessary.
- Fixed some indending for readability.
- Removed three unused variables (`audio_arrays`, `has_audio`, and `has_text`).

I reran `model run inference.py` thrice and got the following results:

    ```txt
    AGGREGATE PERFORMANCE METRICS:
        Average time to first byte: 
            run1: 1.347s
            run2: 1.336s
            run3: 1.590s
        Average realtime factor:
            run1: 0.849
            run2: 0.813
            run3: 1.059
    ```

    TTFB            = ~(1.347 + 1.336 + 1.590)/3 = ~1.424s
    Realtime Factor = ~(0.849 + 0.813 + 1.059)/3 = ~0.907

    TTFB: Improved from 1.590s to 1.424s (-10.4%)
    Realtime Factor: Improved from 1.048 to 0.907 (-13.5%)

Clearly this is not due to my afformentioned cleanup, and the difference from my last run (TTFB +2.5% & Realtime Factor -10.9%) is due to noise. It is noteworthy nonetheless.

## KV-cache Improvements

I then reread the project with more care and research and made the following observation.

- KV-cache optimizations (static cache already implemented)
    - From profiling_1.txt
        > We detected that you are passing `past_key_values` as a tuple and this
        > is deprecated and will be removed in v4.43. Please use an appropriate
        > `Cache` class
        > (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)
        > The `seen_tokens` attribute is deprecated and will be removed in
        > v4.41. Use the `cache_position` model input instead. 
    - Should use HuggingFace `Cache` classes instead of depreciated tuple format.
        - `transformers.StaticCache` or `transformers.DynamicCache`
            - 512 - 1024 for StaticCache
        - Use `cache_position` instead of `seen_tokens`
        - Could have an effect on flash_attn and worth retrying post-hoc

Upon inspection I realized that `past_key_values` is being passed in the underlying `MiniCPM-o` implementation and is not exposed in the API.

[MiniCPM-o/omnilmm/model/omnilmm.py#L188](https://github.com/OpenBMB/MiniCPM-o/blob/2d9919ac6998e603f9a6fa23f61a82d8e43617f6/omnilmm/model/omnilmm.py#L188)

```python
# ...
class OmniLMMModel(MistralModel):
    # ...
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None, # <- HERE
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
    # ...
```

This should be changed to a `transformers.StaticCache` or `DynamicCache` dtype, and updated accordingly wherever `OmniLMMModel.forward` is called. However, due to time constraints, I did not attempt to alter the package or PR a contrib. Might later.

## Audio Buffering Improvements

Currently in `MinicpmInferenceEngine.run` we are extending a Python list as each audio chunk comes in. This could be improved by pre-allocating a numpy array.

This was done with care taken to overflow scenarios (which did not present) and the following runs were executed:

```txt
AGGREGATE PERFORMANCE METRICS:
    Average time to first byte: 
        run1: 1.322s
        run2: 1.356s
        run3: 1.402s
    Average realtime factor:
        run1: 0.827
        run2: 0.830
        run3: 0.869
```

TTFB            = ~(1.322 + 1.356 + 1.402)/3 = ~1.360s
Realtime Factor = ~(0.827 + 0.830 + 0.869)/3 = ~0.842

TTFB: Improved from 1.590s to 1.360s (-14.5%)
Realtime Factor: Improved from 1.048 to 0.842 (-19.7%)