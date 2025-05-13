import matplotlib.pyplot as plt
import torch


def visualize_sample(model, dataset, device):
    model.eval()
    image, mask = dataset[0]
    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(device)).squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image.squeeze(), cmap='gray'); axs[0].set_title('Image')
    axs[1].imshow(mask.squeeze(), cmap='gray'); axs[1].set_title('Ground Truth')
    axs[2].imshow(pred > 0.5, cmap='gray'); axs[2].set_title('Prediction')
    plt.show()

def main():
    pass

if __name__ == '__main__':
    main()
