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
        from encodec import EncodecModel
        from encodec.utils import convert_audio
    except ImportError:
        print("Error: 'encodec' is not installed. Please install it via: pip install encodec")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.to(device)

    # Load and convert audio
    wav, sr = torchaudio.load(args.input)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0).to(device)

    # Encode and Decode
    with torch.no_grad():
        encoded_frames = model.encode(wav)
        decoded = model.decode(encoded_frames)[0]

    # Save output
    decoded = decoded.squeeze(0).cpu()
    torchaudio.save(args.output, decoded, model.sample_rate)

if __name__ == "__main__":
    main()
