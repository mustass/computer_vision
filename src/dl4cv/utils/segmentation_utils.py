import matplotlib.pyplot as plt
import numpy as np
import os
from torch.nn.functional import sigmoid
import warnings
warnings.filterwarnings("ignore")

def plot_results(preds, labels, images, epoch, batch_index , batch_size, dir_path, wandb_logger, test=False):
    """
    A function to plot the results of the segmentation
    and save them in the output folder of the run
    """

    # Define ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])


    plt.rcParams["figure.figsize"] = [15, 10]
    preds = sigmoid(preds)
    f, axarr = plt.subplots(4,batch_size)
    for k in range(batch_size):    
        img = images[k].cpu().numpy()
        img = np.rollaxis(img, 0, 3)
        img = img*std+mean
        axarr[0,k].imshow(img)
        axarr[0,k].title.set_text("Input")
        axarr[0,k].axis("off")
        axarr[1,k].imshow(preds[k].cpu().numpy(), cmap="gray")
        axarr[1,k].title.set_text("Output")
        axarr[1,k].axis("off")
        axarr[2,k].imshow(preds[k].cpu().numpy()>0.5, cmap="gray")
        axarr[2,k].title.set_text("Prediction")
        axarr[2,k].axis("off")
        axarr[3,k].imshow(labels[k].cpu().numpy(), cmap="gray")
        axarr[3,k].title.set_text("True")
        axarr[3,k].axis("off")

        if test:
            os.makedirs(dir_path + "/plots",exist_ok=True)
            plt.imsave(dir_path + f'/plots/b_{batch_index}_idx_{k}.png',preds[k].cpu().numpy(), cmap="gray")
    

    string = f"validation_e_{epoch}_b_{batch_index}" if not test else f"test_e_{epoch}_b_{batch_index}"

    if not os.path.exists(dir_path + "/plots"):
        os.makedirs(dir_path + "/plots")
    plt.savefig(dir_path + f"/plots/plots_{string}.png")
    wandb_logger.log_image(key='plots', images = [dir_path + f"/plots/plots_{string}.png"],caption = [string])
    os.remove(dir_path + f"/plots/plots_{string}.png")