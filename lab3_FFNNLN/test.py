from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import random

dir_path = './logs'

try:
    shutil.rmtree(dir_path)
except OSError as e:
    print("Error: %s : %s" % (dir_path, e.strerror))

if not os.path.exists("./logs"):
    os.makedirs("./logs")

writer = SummaryWriter('./logs')

for epoch in range(100):
    train_loss = epoch**2+3*epoch + 5*random.uniform(0,100)
    writer.add_scalar("loss/train", train_loss, epoch)

writer.close()
