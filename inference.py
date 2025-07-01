from pathlib import Path

import modal
import numpy as np
import soundfile as sf

import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
# Save Hugging Face Token
HF_TOKEN = os.getenv('HF_TOKEN')


MODEL_REVISION = "9da79acdd8906c7007242cbd09ed014d265d281a"


app = modal.App(name="minicpm-inference-engine")


minicpm_inference_engine_image = (
    # install MiniCPM-o dependencies
    modal.Image.from_registry(f"nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    # Install flash-attn dependencies
    .pip_install(  # required to build flash-attn
        "ninja==1.11.1.3",
        "packaging==24.2",
        "wheel",
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "torchvision==0.22.1",
    )
    .run_commands(  # add flash-attn
        "pip install --upgrade flash_attn==2.8.0.post2"
    )
    # Install general AI dependencies
    .pip_install(
        "huggingface_hub[hf_transfer]==0.30.1",
        "transformers==4.44.2",
        "onnxruntime==1.20.1",
        "scipy==1.15.2",
        "numpy==1.26.4",
        "pandas==2.2.3",
    ).pip_install(
        "Pillow==10.1.0",
        "sentencepiece==0.2.0",
        "vector-quantize-pytorch==1.18.5",
        "vocos==0.1.0",
        "accelerate==1.2.1",
        "timm==0.9.10",
        "soundfile==0.12.1",
        "librosa==0.9.0",
        "sphn==0.1.4",
        "aiofiles==23.2.1",
        "decord",
        "moviepy",
        "pydantic",
    )
    .pip_install("gekko")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": "/cache",
            "HF_TOKEN": HF_TOKEN,
        }
    )  # and enable it
    .add_local_python_source("minicpmo")
)


with minicpm_inference_engine_image.imports():
    import time
    from minicpmo import MiniCPMo, AudioData
    import numpy as np
    

MODAL_GPU = "A10G"

@app.cls(
    cpu=2,
    memory=5000,
    gpu=MODAL_GPU,
    image=minicpm_inference_engine_image,
    min_containers=1,
    timeout=15 * 60,
    volumes={
        "/cache": modal.Volume.from_name("huggingface-cache", create_if_missing=True),
    },
)
class MinicpmInferenceEngine:
    @modal.enter()
    def load(self):
        print("Starting model loading...")
        load_start = time.perf_counter()

        # time model initialization
        model_init_start = time.perf_counter()
        self.model = MiniCPMo(device="cuda", model_revision=MODEL_REVISION)
        model_init_time = time.perf_counter() - model_init_start

        # time TTS initialization
        tts_init_start = time.perf_counter()
        self.model.init_tts()
        tts_init_time = time.perf_counter() - tts_init_start

        total_load_time = time.perf_counter() - load_start

        print(f"LOAD TIMING:")
        print(f"   Model initialization: {model_init_time:.3f}s")
        print(f"   TTS initialization: {tts_init_time:.3f}s")
        print(f"   Total load time: {total_load_time:.3f}s")


    @modal.method()
    def run(self, text: str):
        print(f"\nStarting inference for: '{text}'")
        inference_start = time.perf_counter()

        audio_data = []
        time_to_first_byte = None
        time_to_first_audio = None
        audio_processing_time = 0

        for item in self.model.run_inference([text]):
            if item is None:
                break
            if isinstance(item, str):
                print(f"Got text from MiniCPM: {text}")
            if isinstance(item, AudioData):
                assert item.sample_rate == 24000

                if time_to_first_byte is None:
                    time_to_first_byte = time.perf_counter() - inference_start
                    print(f"   First byte received at: {time_to_first_byte:.3f}s")
                
                if time_to_first_audio is None:
                    time_to_first_audio = time.perf_counter() - inference_start
                    print(f"   First audio received at: {time_to_first_audio:.3f}s")
                    
                # time audo processing
                audio_proc_start = time.perf_counter()
                audio_data.append(item.array)
                audio_processing_time += time.perf_counter() - audio_proc_start

        generation_time = time.perf_counter() - inference_start

        if len(audio_data) == 0:
            raise ValueError("No audio data received")
        
        # time final audio concat
        concat_start = time.perf_counter()
        full_audio = np.concatenate(audio_data)
        concat_time = time.perf_counter() - concat_start

        total_time = time.perf_counter() - inference_start
        audio_duration = len(full_audio) / 24000
        realtime_factor = total_time / audio_duration if audio_duration > 0 else float('inf')

        print(f"INFERENCE TIMING:")
        print(f"   Time to first byte: {time_to_first_byte:.3f}s")
        print(f"   Time to first audio: {time_to_first_audio:.3f}s")
        print(f"   Generation time: {generation_time:.3f}s")
        print(f"   Audio processing time: {audio_processing_time:.9f}s")
        print(f"   Audio concat time: {concat_time:.9f}s")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Audio duration: {audio_duration:.3f}s")
        print(f"   Real-time factor: {realtime_factor:.3f}")

        return {
            "time_to_first_byte": time_to_first_byte,
            "time_to_first_audio": time_to_first_audio,
            "generation_time": generation_time,
            "audio_processing_time": audio_processing_time,
            "concat_time": concat_time,
            "total_time": total_time,
            "audio_array": full_audio,
            "sample_rate": 24000,
            "text": text,
            "audio_duration": audio_duration,
            "realtime_factor": realtime_factor,
        }
    


@app.local_entrypoint()
def main():
    engine = MinicpmInferenceEngine()

    # Warmup
    print("Warming up...")
    result = engine.run.remote("Hi, how are you?")

    results = []
    for text in ["I'm fine, thank you!", "What's your name?", "My name is John Doe", "What's your favorite color?", "My favorite color is blue", "What's your favorite food?", ]:
        result = engine.run.remote(text)
        results.append(result)

    PARENT_DIR = Path(__file__).parent

    # process results and collect timing data
    ttfb_times = []
    realtime_factors = []
    generation_times = []
    audio_proc_times = []

    for result in results:
        sf.write(PARENT_DIR / f"{result['text']}.wav", result["audio_array"], result["sample_rate"])
        print(f"Wrote {result['text']}.wav to {PARENT_DIR / result['text']}.wav")

        ttfb_times.append(result["time_to_first_byte"])
        realtime_factors.append(result["realtime_factor"])
        generation_times.append(result["generation_time"])
        audio_proc_times.append(result["audio_processing_time"])

    print(f"AGGREGATE PERFORMANCE METRICS:")
    print(f"   Average time to first byte: {np.mean(ttfb_times):.3f}s")
    print(f"   Average realtime factor: {np.mean(realtime_factors):.3f}")
    print(f"   Average generation time: {np.mean(generation_times):.3f}s")
    print(f"   Average audio processing time: {np.mean(audio_proc_times):.3f}s")
    print(f"   TTFB std dev: {np.std(ttfb_times):.3f}s")
    print(f"   Realtime factor std dev: {np.std(realtime_factors):.3f}")




