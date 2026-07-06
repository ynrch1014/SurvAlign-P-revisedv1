import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input wav file path")
    parser.add_argument("--output", required=True, help="Output wav file path")
    args = parser.parse_args()

    try:
        from clearvoice import ClearVoice
    except ImportError:
        print("Error: 'clearvoice' is not installed. Please install it via: pip install clearvoice")
        sys.exit(1)

    # Note: clearvoice will automatically download the model to cache on first run
    try:
        myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])
        output_wav = myClearVoice(input_path=args.input, online_write=False)
        myClearVoice.write(output_wav, output_path=args.output)
    except Exception as e:
        print(f"Error running ClearVoice: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
