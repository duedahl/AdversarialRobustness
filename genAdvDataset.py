import torch, pickle, os, argparse
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import ViTImageProcessor, ViTForImageClassification
import torchvision.transforms as transforms
from torchvision import datasets
from captum.robust import FGSM, MinParamPerturbation
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

import loadVim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fgsmAttack(inputs, model, true_label, epsilon=0.1):
  # Wrapper of the model that outputs logits - needed for FGSM.perturb
  forward_func = lambda *args, **kwargs: model(*args, **kwargs).logits
  fgsm = FGSM(forward_func, lower_bound=-1, upper_bound=1)
  return fgsm.perturb(inputs, epsilon=epsilon, target=true_label)

def calcImageDistanceMetrics(im1, im2):
    # L-infinity
    linf = torch.norm(im1 - im2, p=float('inf')).item()
    # SSIM and MSE
    im1 = im1.squeeze(0).cpu().numpy().transpose((1, 2, 0))
    im2 = im2.squeeze(0).cpu().numpy().transpose((1, 2, 0))
    mse = mean_squared_error(im1, im2)
    ssim_val = ssim(im1, im2, data_range=(im2.max() - im2.min()), multichannel=True, channel_axis=2)
    return mse, ssim_val, linf

def saveImageTensor(tensor, path, label, fname):
    # Saving as Image and Re-loading causes small loss +-0.001, but that is enough to cause the attack to fail
    # As such, we save the tensors and load them as a dataset.

    # Ensure directory exists
    if isinstance(label, str):
        dir = f"{path}/{label}"
    else:
        dir = f"{path}/{label.item()}"
    os.makedirs(dir, exist_ok=True)
    torch.save(tensor, f"{dir}/{fname}.pt")

def genImageNetteConvDict(model):
    # Construct dict to translate from Imagenette labels to ImageNet labels

    labels = "tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute"
    labels = labels.split(", ")

    netteToImageNetConv = {}
    for i, newlabel in enumerate(labels):
        for key, val in model.config.label2id.items():
            # Check which entry in labels is contained in the key
            index = key.find(newlabel)
            if index != -1:
                netteToImageNetConv[i] = val

    return netteToImageNetConv

def main():

    # Get dataset and path from command line
    parser = argparse.ArgumentParser(description='Generate adversarial dataset')
    parser.add_argument('--dataset', type=str, default='imagenette', help='Dataset to use')
    parser.add_argument('--path', type=str, default='./data_adv', help='Path to generate dataset')
    parser.add_argument('--model', type=str, default='google/vit-base-patch16-224', help='Model to use')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load')
    parser.add_argument('--blur', action='store_true', help='Apply blur to images')
    args = parser.parse_args()

    # Load Model and prepare processor    
    if args.checkpoint:
        if args.model == 'google/vit-base-patch16-224':
            model, processor = loadVim.prepareDownstreamVit()
        else:
            model, processor = loadVim.prepareDownstreamResnet()
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = AutoModelForImageClassification.from_pretrained(args.model)
        processor = AutoImageProcessor.from_pretrained(args.model)
       
    # Update label mapping
    dirs = sorted([d for d in os.listdir(args.dataset)])
    id2label = {key: val for key, val in enumerate(dirs)}
    model.config.id2label = id2label
        
    model.to(device)
    model.eval()
    torch.no_grad()
    processor.do_normalize = False  # Strip preprocessing except resize
    normalize = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    netteToImageNetConv = genImageNetteConvDict(model)

    # Predict function
    def pred(inputs, model, preprocess=True):
        # preprocess the image
        if preprocess:
            inputs = normalize(inputs)

        # If input is not batched, batch it
        if len(inputs.shape) < 4:
            inputs = inputs.unsqueeze(0)

        outputs = model(inputs.to(device))
        logits = outputs.logits
        val, idx = torch.max(logits, dim=1)
        return idx

    # Load dataset
    if args.blur:
        def process_image(image):
            img = processor(image, return_tensors="pt").pixel_values.squeeze(0)
            return transforms.GaussianBlur(kernel_size=11, sigma=2)(img)
    else:
        def process_image(image):
            return processor(image, return_tensors="pt").pixel_values.squeeze(0)
    
    data = datasets.ImageFolder(args.dataset, transform=transforms.Lambda(process_image))

    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

    # Adversarial attack
    # Min Param Perturbation works only with batch size 1 for our purposes
    min_pert = MinParamPerturbation(
                                    forward_func = lambda *args, **kwargs: model(*args, **kwargs).logits,
                                    attack = fgsmAttack,
                                    arg_name = 'epsilon',
                                    mode = "binary",
                                    arg_min = 0.001,
                                    arg_max = 3,
                                    arg_step = 0.001,
                                    preproc_fn = normalize,
                                    apply_before_preproc=True
                                )

    alt_im, min_eps, inputs = None, None, None

    path_adversarial_dataset = args.path
    os.makedirs(path_adversarial_dataset, exist_ok=True)

    log = []
    advFailCount = 0
    
    for i, (input, label) in enumerate(dataLoader):
        input = input.to(device)
        # Convert label item to ImageNet label
        label = label.to(device)

        prediction = pred(input, model)

        if prediction == label:
            # We only care for adversarial examples in the case when the model could correctly predict otherwise
            attack_kwargs={'model':model,'true_label':label}

            alt_im, min_eps = min_pert.evaluate(input, attack_kwargs=attack_kwargs, target=label)
            
            if alt_im is None:
                print("Could not generate adversarial example")
                advFailCount += 1
                continue

            altpred = pred(alt_im, model)
            print(f"adv_{i} is a {model.config.id2label[label.item()]} but classifies as {model.config.id2label[altpred.item()]} with epsilon {min_eps}")

            mse, ssim_val, linf = calcImageDistanceMetrics(input, alt_im)
            log.append({"index": i, "label": label.item(), "prediction": altpred.item(), "epsilon": min_eps, "mse": mse, "ssim": ssim_val, "linf": linf})

            # Save the adversarial example to new dataset
            saveImageTensor(alt_im, path_adversarial_dataset, model.config.id2label[label.item()], f"adv_{i}")
        else:
            print(f"Incorrect classification, label: {label}, prediction: {prediction}, skipping...")

    # Pickle the log
    with open(f"{path_adversarial_dataset}/log.pkl", "wb") as f:
        pickle.dump({"fails":advFailCount, "log": log}, f)

if __name__ == "__main__":
    main()