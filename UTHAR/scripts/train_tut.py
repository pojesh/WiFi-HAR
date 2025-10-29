import torch
from torch.utils.data import DataLoader
from dataloader.ut_har_dataset import UTHARDataset
from models.transformer_tut import TemporalUncertaintyTransformer

def train(cfg):
    trainset = UTHARDataset(cfg['x_train'], cfg['y_train'], corruption_level=cfg['corruption'], jitter=cfg['jitter'])
    valset = UTHARDataset(cfg['x_val'], cfg['y_val'], corruption_level=0.0)
    train_loader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(valset, batch_size=cfg['batch_size'])

    model = TemporalUncertaintyTransformer(num_classes=cfg['num_classes']).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(cfg['epochs']):
        model.train()
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        if acc > best_acc:
            torch.save(model.state_dict(), 'results/tut_best.pth')
            best_acc = acc
        print(f'Epoch {epoch}, Val acc: {acc:.4f}')

if __name__ == '__main__':
    import yaml
    with open('configs/config_tut.yaml','r') as f:
        cfg = yaml.safe_load(f)
    train(cfg)
