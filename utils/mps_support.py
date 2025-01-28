import torch

def check_mps_support():
    if torch.backends.mps.is_available():
        print("MPS is available!")
        print(f"Using MPS device: {torch.device('mps')}")
        return True
    else:
        print("MPS is not available.")
        if torch.backends.mps.is_built():
            print("MPS backend is built, but your system might not support it.")
        else:
            print("MPS backend is not built.")
        return False

if __name__ == "__main__":
    check_mps_support()