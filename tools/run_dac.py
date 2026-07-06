import argparse
import sys
import torch
import torchaudio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input wav file path")
    parser.add_argument("--output", required=True, help="Output wav file path")
    args = parser.parse_args()

    try:
        import dac
        from audiotools import AudioSignal
    except ImportError:
        print("Error: 'dac' or 'audiotools' is not installed. Please install via: pip install descript-audio-codec audiotools")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Download and load model
    model_path = dac.utils.download(model_type="16khz")
    model = dac.DAC.load(model_path).to(device)
    model.eval()

    # Load audio
    signal = AudioSignal(args.input)
    signal.to(model.device)
    
    # Encode and Decode
    with torch.no_grad():
        x = model.preprocess(signal.audio_data, signal.sample_rate)
        z, codes, latents, _, _ = model.encode(x)
        y = model.decode(z)

    # Save output
    signal = signal.clone()
    signal.audio_data = y
    signal.write(args.output)

if __name__ == "__main__":
    main()
