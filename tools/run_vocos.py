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
        from vocos import Vocos
    except ImportError:
        print("Error: 'vocos' is not installed. Please install it via: pip install vocos")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Vocos encodec model
    vocos = Vocos.from_pretrained("charactrix/vocos-encodec-24khz").to(device)

    try:
        from encodec import EncodecModel
        from encodec.utils import convert_audio
    except ImportError:
        print("Error: 'encodec' is required for this Vocos model. Please install via: pip install encodec")
        sys.exit(1)

    encodec = EncodecModel.encodec_model_24khz()
    encodec.set_target_bandwidth(6.0)
    encodec.to(device)

    wav, sr = torchaudio.load(args.input)
    wav = convert_audio(wav, sr, encodec.sample_rate, encodec.channels)
    wav = wav.unsqueeze(0).to(device)

    with torch.no_grad():
        encoded_frames = encodec.encode(wav)
        features = vocos.codes_to_features(encoded_frames[0][0])
        decoded = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

    decoded = decoded.cpu()
    torchaudio.save(args.output, decoded, 24000)

if __name__ == "__main__":
    main()
