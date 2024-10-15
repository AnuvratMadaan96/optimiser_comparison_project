import torch.optim as optim

def get_optimizer(optimizer_name, model, lr):
    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f'Optimizer {optimizer_name} not recognized')