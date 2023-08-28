import numpy as np
import torch
import deepinv as dinv
from torchvision import datasets, transforms
from fastmri.data import subsample


def get_params():
    params = 20
    return params


def get_unrolled(img_size, device):
    prior = dinv.optim.PnP(dinv.models.DnCNN(img_size[0], img_size[0], depth=7, train=True, pretrained=None).to(device))
    model = dinv.unfolded.unfolded_builder(iteration="HQS", prior=prior, data_fidelity=dinv.optim.data_fidelity.L2(),
                                         max_iter=5, params_algo={"stepsize": 1., "g_param": 0.1, "lambda": 1.})
    return model


def get_dataset(operator, test_train=False):
    if operator == 'Tomography':
        tfs = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128)), transforms.Grayscale(1)])
    else:
        tfs = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))])
    # dataset = datasets.MNIST(root='/projects/UDIP/deepinv/datasets/', train=False, transform=transforms.ToTensor())

    if test_train:
        return [datasets.CelebA(root='/projects/UDIP/deepinv/datasets/', split="train", transform=tfs), datasets.CelebA(root='/projects/UDIP/deepinv/datasets/', split="test", transform=tfs)]
    else:
        return datasets.CelebA(root='/projects/UDIP/deepinv/datasets/', split="test", transform=tfs)


def find_operator(name, params, k, img_size, device):
    if name == "CS":
        param = int(np.logspace(-1, 0, params)[params-k-1]*np.prod(img_size))
        p = dinv.physics.CompressedSensing(m=param, img_shape=img_size, device=device)
        code = f'physics = dinv.physics.CompressedSensing(m={param}, fast=False, ...)'
    elif name == "fastCS":
        param = int(np.logspace(-1, 0, params)[params-k-1]*np.prod(img_size))
        p = dinv.physics.CompressedSensing(
            m=param, fast=True, channelwise=False, img_shape=img_size, device=device
        )
        code = f'physics = dinv.physics.CompressedSensing(m={param}, fast=True, ...)'
    elif name == "inpainting":
        param = np.logspace(-1, 0, params)[params-k-1]
        p = dinv.physics.Inpainting(tensor_size=img_size, mask=param, device=device)
        code = f'physics = dinv.physics.Inpainting(mask={param:.2f}, ...)'
    elif name == "MRI":
        param = int(k/2)+1

        # Create a mask function
        mask_func = subsample.RandomMaskFunc(
            center_fractions=[0.32/param],
            accelerations=[param]
        )
        m = mask_func.sample_mask((img_size[1], img_size[0]), offset=None)
        mask = torch.ones((img_size[0], 1)) * (m[0] + m[1]).permute(1, 0)
        mask[mask > 1] = 1

        p = dinv.physics.MRI(mask=torch.ones(img_size[-2], img_size[-1]), device=device)
        code = f'physics = dinv.physics.MRI(mask=torch.ones({img_size[-2]}, {img_size[-1]}), ...)'
    elif name == "Tomography":
        param = img_size[-1] - k*6
        p = dinv.physics.Tomography(
            img_width=img_size[-1], angles=param, device=device
        )
        code = f'physics = dinv.physics.Tomography(angles={param}, ...)'
    elif name == "denoising":
        param = np.logspace(-1, 0, params)[params-k-1]
        p = dinv.physics.Denoising(dinv.physics.GaussianNoise(param))
        code = f'physics = dinv.physics.Denoising(dinv.physics.GaussianNoise({param:.2f}))'
    elif name == "pansharpen":
        param = int(k/2)+1
        p = dinv.physics.Pansharpen(img_size=img_size, factor=param, device=device)
        code = f'physics = dinv.physics.Pansharpen(factor={param}, ...)'
    elif name == "fast_singlepixel":
        param = int(np.logspace(-1, 0, params)[params-k-1]*np.prod(img_size))
        p = dinv.physics.SinglePixelCamera(
            m=param, fast=True, img_shape=img_size, device=device
        )
        code = f'physics = dinv.physics.SinglePixelCamera(m=f{param}, fast=True, ...)'
    elif name == "singlepixel":
        param = int(np.logspace(-1, 0, params)[params-k-1]*np.prod(img_size))
        p = dinv.physics.SinglePixelCamera(
            m=param, fast=False, img_shape=img_size, device=device
        )
        code = f'physics = dinv.physics.SinglePixelCamera(m=f{param}, fast=False, ...)'
    elif name == "deblur_fft":
        param = k/params
        p = dinv.physics.BlurFFT(
            img_size=img_size,
            filter=dinv.physics.blur.gaussian_blur(sigma=param),
            device=device,
        )
        code = f'physics = dinv.physics.BlurFFT(filter=dinv.physics.blur.gaussian_blur(sigma={param:.1f}), ...)'
    elif name == "super_resolution":
        param = int(k/2)+1
        p = dinv.physics.Downsampling(img_size=img_size, factor=param, device=device)
        code = f'physics = dinv.physics.Downsampling(factor={param}, ...)'
    else:
        raise Exception("The inverse problem chosen doesn't exist")
    return p, code, param
