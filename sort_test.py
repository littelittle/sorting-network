import torch 
from models.toy_model import sort_model
from utils.config import load_config
# from config.toy_config import toy_config as config


config = load_config("config/toy_config.yaml")


# check cuda available
if torch.cuda.is_available() is True:
    device = "cuda"
print(f"using device {device}")


# load model checkpoints
checkpoint = torch.load("checkpoints/checkpoint_epoch_100.pth", weights_only=True)
model = sort_model(config['model'])
model.load_state_dict(checkpoint['model_state_dict'])


# test the model
unsorted = torch.randint(0, 100, (config['model']["input_size"], )).float()
gt, _ = torch.sort(unsorted)
print(f"groundtruth:{gt.int()}")
sorted = model(unsorted)
print(f"sorted:{sorted.int()}")
print(f"mean error:{torch.abs(gt-sorted).mean()}")
