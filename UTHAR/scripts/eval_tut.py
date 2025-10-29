# eval_tut.py
import torch
from dataloader.ut_har_dataset import UTHARDataset
from models.transformer_tut import TemporalUncertaintyTransformer

def evaluate(cfg):
    corruption_levels = cfg['corruption_levels']
    model = TemporalUncertaintyTransformer(num_classes=cfg['num_classes']).cuda()
    model.load_state_dict(torch.load('results/tut_best.pth'))

    results = []
    for corr in corruption_levels:
        testset = UTHARDataset(cfg['x_test'], cfg['y_test'], corruption_level=corr)
        test_loader = DataLoader(testset, batch_size=cfg['batch_size'])
        correct, total = 0, 0
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        acc = correct / total
        results.append({'corruption': corr, 'accuracy': acc})
        print(f'Corruption: {corr}, Accuracy: {acc:.4f}')
    # Save results
    import pandas as pd
    pd.DataFrame(results).to_csv('results/tut_results.csv', index=False)

if __name__ == '__main__':
    import yaml
    from torch.utils.data import DataLoader
    with open('configs/config_tut.yaml','r') as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg)
