import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# from config.toy_config import toy_config
from utils.config import load_config
from models.toy_model import sort_model
from models.SortTrans import seq_transformer, seq_encoder_only
from datasets.toy_dataset import sort_dataset

# config = load_config("config/toy_config.yaml")
config = load_config("config/SortEncoder.yaml")

device = torch.device("cuda" if (torch.cuda.is_available() and config['device']) else "cpu")
print(f"using device: {device}(printed once)")

# model = sort_model(config=config["model"]).to(device)
model = seq_encoder_only(config["model"]).to(device)
dataset = sort_dataset(config=config)

# load the checkpoints
base_epoch = 0
if os.path.exists(config["checkpoints_path"]) and len(os.listdir(config["checkpoints_path"])):
    load_path = os.path.join(config["checkpoints_path"], os.listdir(config["checkpoints_path"])[-1])
    print("loading checkpoints:"+load_path)
    name = load_path.split('.')[0]
    base_epoch = int(name.split('-')[-1])
    checkpoints = torch.load(os.path.join(config["checkpoints_path"], os.listdir(config["checkpoints_path"])[-1]), weights_only=True)
    model.load_state_dict(checkpoints['model_state_dict'])

# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
writer = SummaryWriter(f'experiments/{config["experiment_name"]}')
dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], num_workers=0)

def train_model(config):
    for epoch in range(base_epoch, config["training"]["epochs"]+base_epoch):
        running_loss = 0.0
        for inputs, targets in dataloader:
            model.train()
            logits, _ = model(inputs.to(device).float())
            loss = criterion(logits, targets.view(-1).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        writer.add_scalar("Loss/train", running_loss/len(dataloader), epoch)

        if config["evaluation"]["eval_enabled"]:
            model.eval()
            with torch.no_grad():   
                eval_unsorted = torch.randint(0, 100, (config["evaluation"]["eval_batch_size"], config["model"]["input_size"])).to(device).float()
                sorted, _= torch.sort(eval_unsorted)
                _, predicted = model(eval_unsorted)
                residual = torch.abs(sorted-predicted).mean()
                writer.add_scalar("MeanError/train", residual, epoch)
                print("epoch:{}\t| loss:{}\t| residual:{}".format(epoch, running_loss/len(dataloader), residual))

        if (epoch+1)%100 == 0:
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            save_path = config["checkpoints_path"]+f'checkpoint_epoch_{epoch+1}.pth'
            if not os.path.exists(config["checkpoints_path"]):
                os.mkdir(config["checkpoints_path"])
            torch.save(checkpoint, save_path)
            print(f'Checkpoint saved at epoch {epoch+1}')


def eval_model():
    pass

if __name__ == "__main__":
    train_model(config)

writer.flush()
writer.close()

