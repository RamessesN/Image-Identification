#########################   Instruction   ##########################
#####                     Training  Model                       ####
####################################################################

import mlx.core as mx
import mlx.optimizers as optim
from model import CNN
from pathlib import Path
from data_utils import load_dataset
from loss_utils import cross_entropy_loss

def train_model():
    model = CNN()

    dataset = load_dataset(Path("./data"), 64)

    optimizer = optim.Adam(learning_rate=0.000001)

    def loss_fn(params, images, labels):
        model.update(params)  # Update model parameters
        images = mx.transpose(images, (0, 2, 3, 1))  # 调整输入形状
        outputs = model(images)
        return cross_entropy_loss(outputs, labels)
    loss_and_grad_fn = mx.value_and_grad(loss_fn)

    num_epochs = 5000
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataset:
            images, labels = batch["image"], batch["label"]
            params = model.parameters()
            loss, grads = loss_and_grad_fn(params, images, labels)
            optimizer.update(model, grads)
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

    print("Training finished！")
    return model