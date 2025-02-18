import torch.nn.functional as F

# Train Function
def train_model(model, device, train_dataloader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
    if epoch % log_interval == 0:
        print(f'Train Epoch: {epoch}\tLoss: {loss.item():.6f}')
