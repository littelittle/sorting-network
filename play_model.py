import time
import torch
import matplotlib.pyplot as plt
from utils.config import load_config
from models.SortTrans import seq_transformer, seq_encoder_only

verbose = False
config = load_config("config/SortEncoder.yaml")

# check cuda available
if torch.cuda.is_available() is True:
    device = "cuda"
else:
    device = "cpu"
config["device"] = device
print(f"using device {device}")

# load model checkpoints
epoch = 100
checkpoint = torch.load(config['checkpoints_path']+f'checkpoint_epoch_{epoch}.pth', weights_only=True)
model = seq_encoder_only(config['model']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

def play():
    # paly the model
    sortinglist = torch.randint(0, 100, (128, 10))
    # sortinglist = [[48, 33, 83, 60,  0, 46, 57, 82, 82, 80,]]
    # sortinglist = torch.tensor(sortinglist)
    # for i in range(10):
    #      sortinglist.append(int(input(f"enter the {i} th number: ")))


    sortinglist = torch.tensor(sortinglist).to(device)
    with torch.no_grad():
        model.eval()
        logits, sortedlist = model(sortinglist)

    # plot the distribution
    # Reshape logits to combine batch and sequence dimensions
    batch_size,  n_vocab = logits.shape
    flattened_logits = logits.reshape(-1, n_vocab)

    # Convert logits to probabilities using softmax
    probabilities = torch.nn.functional.softmax(flattened_logits, dim=-1)

    if verbose: 
        for i in range(10):
            # Select one distribution to visualize (e.g., first position of first batch)
            selected_dist = probabilities[i].cpu().numpy()

            # Create bar plot
            plt.figure(figsize=(10, 6))
            plt.bar(range(n_vocab), selected_dist)
            plt.xlabel('Token Value')
            plt.ylabel('Probability')
            plt.title('Distribution for First Position')
            plt.show()

        print(f"sorting list is {sortinglist.detach().cpu().numpy()}")
        print(f"sorted list is {sortedlist.detach().cpu().numpy()}")


if __name__ == "__main__":
    start_time = time.time()
    for i in range(1000):
        play()
    print(f"with {device}, total cost is {time.time()-start_time}s")

