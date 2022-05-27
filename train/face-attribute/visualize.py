
import torch
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_result(args, model, test_loader):
    images, labels = next(iter(test_loader))
    model = model.to(device)
    preds = torch.round(model(images.to(device))).cpu().detach()
    images = torch.permute(images, (0, 2, 3, 1))
    plt.figure(figsize=(10, 50))
    for i in range(10):
        plt.subplot(10, 1, i+1)
        plt.imshow(images[i])
        plt.title(f" Actual:  {labels[i]}\nPredict: {preds[i].to(torch.int)}")
        plt.axis('off')
        plt.savefig(args.pred)
    plt.show()



def plot_pred(args, model, test_loader):
    images = next(iter(test_loader))
    model = model.to(device)
    preds = torch.round(model(images.to(device))).cpu().detach()
    images = torch.permute(images, (0, 2, 3, 1))
    plt.figure(figsize=(10, 50))
    for i in range(10):
        plt.subplot(10, 1, i+1)
        plt.imshow(images[i])
        plt.title(f"{preds[i].to(torch.int)}")
        plt.axis('off')
        plt.savefig(args.pred)
    plt.show()
