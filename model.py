import torch 
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class Configt: 
    cw_size : int = 8
    batch_size : int = 4

def train_test_model():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
        print(text)
def main():
    train_test_model()

if __name__ == "__main__":
    main()
