import torch
from manim import *
import deepinv as dinv
from helper_funcs import find_operator, get_dataset, get_params, get_unrolled
import os

def torch2np(x):
    return np.uint8(255*x[0, :, :, :].detach().permute(1, 2, 0).cpu().numpy())

total_slide_time = 20 # in seconds


def code(operator_string, methods):
    code = '''
    import deepinv as dinv
    
    ''' + operator_string + '''
    y = physics(x)
    
    '''
    if "linear" in methods:
        code += '''x_lin = physics.A_adjoint(y)
    '''
    if "plug-and-play" in methods:
        code += '''model = dinv.optim.optim_builder("PGD", max_iter=2e3, ...)
    '''
    if "unfolded" in methods:
        code += '''model = dinv.unfolded.unfolded_builder("PGD", max_iter=4, ...)
    '''
    if "diffusion" in methods:
        code += '''model = dinv.sampling.DDRM(denoiser=DRUNet(...), ...)
    '''
    code += '''modelx(y, physics) '''
    return code


class DeepInvSlide(Scene):
    def add_image(self, x, title, height=3., label_scale=.3):
        img = ImageMobject(torch2np(x))
        img.add(Text(title).scale(label_scale).next_to(img, .2*UP))
        img.height = height
        self.imgs.append(img)

    def my_construct(self, operator, methods, tex_func, img_index=1):
        device = 'cuda:0'

        dataset = get_dataset(operator)
        x = dataset[img_index][0].unsqueeze(0).to(device)

        torch.manual_seed(0)

        img_size = x.shape[1:]
        params = get_params()
        drunet = dinv.models.DRUNet(img_size[0], img_size[0], pretrained='download').to(device)
        dncnn = dinv.models.DnCNN(img_size[0], img_size[0], pretrained='download_lipschitz').to(device)

        for k in range(params):
            physics, operator_code, param = find_operator(operator, params, k, img_size, device)
            y = physics(x)

            height = 8/len(methods)
            label_scale = height/15
            self.imgs = []
            self.add_image(x, "image", height=height, label_scale=label_scale)

            if "measurement" in methods:
                self.add_image(y, "measurement", height=height, label_scale=label_scale)

            if "linear" in methods:
                x_lin = physics.A_adjoint(y)
                self.add_image(x_lin, "linear", height=height, label_scale=label_scale)

            if "plug-and-play" in methods:
                norm = physics.compute_norm(x)
                norm = norm.detach().cpu().numpy()
                prior = dinv.optim.PnP(dncnn)
                lamb = 1.
                pnp_model = dinv.optim.optim_builder(iteration="PGD", prior=prior, data_fidelity=dinv.optim.data_fidelity.L2(),
                                                     max_iter=4000, thres_conv=1e-3, params_algo={"stepsize": 1. / norm/lamb, "g_param": 0.1,
                                                                                "lambda": lamb})
                x_pnp = pnp_model(y, physics)
                self.add_image(x_pnp, "plug-and-play", height=height, label_scale=label_scale)

            if "unfolded" in methods:
                unrolled_model = get_unrolled(img_size, device)
                dir = f"/projects/UDIP/manim/ckpts/{operator}/"

                ckpt = torch.load(dir + os.listdir(dir)[0] + "/ckp_50.pth.tar", map_location=device)
                unrolled_model.load_state_dict(ckpt["state_dict"])
                x_unfold = unrolled_model(y, physics)
                self.add_image(x_unfold, "unfolded", height=height, label_scale=label_scale)

            if "diffusion" in methods:
                diffusion_model = dinv.sampling.DDRM(denoiser=drunet, sigma_noise=.01)
                x_diff = diffusion_model(y + torch.randn_like(y)*.01, physics)
                self.add_image(x_diff, "diffusion", height=height, label_scale=label_scale)

            g = Group(*self.imgs).arrange().shift(2*UP)

            rendered_code = Code(code=code(operator_code, methods), tab_width=4, background="window",
                                 language="Python", font="Monospace").shift(2 * DOWN).scale(0.7)

            tex = Tex(tex_func(param), font_size=32)
            self.add(tex)
            self.add(rendered_code)
            self.add(g)
            self.wait(total_slide_time/params)
            self.remove(g)
            self.remove(tex)
            self.remove(rendered_code)


class Inpainting(DeepInvSlide):
    def construct(self):
        tex_func = lambda param: r"Image inpainting: $y = \text{diag}(m) x$, $m_i\sim \mathcal{B}e(p= " + f"{param:.2f}" + ")$"
        methods = ["measurement", "plug-and-play", "unfolded", "diffusion"]
        self.my_construct("inpainting", methods, tex_func, img_index=11)


class CS(DeepInvSlide):
    def construct(self):
        tex_func = lambda param: r"Compressed Sensing: $y = \begin{bmatrix} I_{" + f"{param}" + r"}  & 0 \end{bmatrix} F\text{diag}{(s)}x$, $s\sim \mathcal{B}e(p=.5)$"
        methods = ["linear", "plug-and-play", "unfolded"]
        self.my_construct("fastCS", methods, tex_func, img_index=14)


class SinglePixelCamera(DeepInvSlide):
    def construct(self):
        tex_func = lambda param: r"Single Pixel Camera: $y = SHx$, $H$ Hadamard transform, $S$ random subsampling."
        methods = ["linear", "plug-and-play", "unfolded"]
        self.my_construct("fast_singlepixel", methods, tex_func, img_index=7)


class SuperResolution(DeepInvSlide):
    def construct(self):
        methods = ["measurement", "plug-and-play", "unfolded"]
        self.my_construct("super_resolution", methods)


class Tomography(DeepInvSlide):
    def construct(self):
        methods = ["measurement", "linear", "plug-and-play", "unfolded"]
        self.my_construct("Tomography", methods)


class MRI(DeepInvSlide):
    def construct(self):
        methods = ["linear", "plug-and-play", "unfolded"]
        self.my_construct("MRI", methods)


class Deblur(DeepInvSlide):
    def construct(self):
        methods = ["measurement", "plug-and-play", "unfolded"]
        self.my_construct("deblur_fft", methods)


class Denoising(DeepInvSlide):
    def construct(self):
        device = 'cuda:0'

        dataset = get_dataset("denoising")
        x = dataset[1][0].unsqueeze(0).to(device)

        torch.manual_seed(0)
        params = get_params()

        img_size = x.shape[1:]
        for k in range(params):
            physics, operator_code, display_y_linear = find_operator("denoising", params, k, img_size, device)

            sigma = physics.noise_model.sigma
            y = physics(x)
            median = dinv.models.MedianFilter()(y, sigma)
            tgv = dinv.models.TGV()(y, sigma*10)
            bm3d = dinv.models.BM3D()(y, sigma)
            drunet = dinv.models.DRUNet(img_size[0], img_size[0], pretrained='download', device=device)(y, sigma)
            imgs = []

            label_scale = .2
            img = ImageMobject(torch2np(x))
            img.add(Text("image").scale(label_scale).next_to(img, .2*UP))
            img.height = 2.4
            imgs.append(img)

            img2 = ImageMobject(torch2np(y))
            img2.add(Text("measurement").scale(label_scale).next_to(img2, .2*UP))
            img2.height = 2.4
            imgs.append(img2)

            img3 = ImageMobject(torch2np(median))
            img3.add(Text("median filter").scale(label_scale).next_to(img3, .2*UP))
            img3.height = 2.4
            imgs.append(img3)

            img4 = ImageMobject(torch2np(tgv))
            img4.add(Text("TGV").scale(label_scale).next_to(img4, .2*UP))
            img4.height = 2.4
            imgs.append(img4)

            img5 = ImageMobject(torch2np(bm3d))
            img5.add(Text("BM3D").scale(label_scale).next_to(img5, .2*UP))
            img5.height = 2.4
            imgs.append(img5)

            img6 = ImageMobject(torch2np(drunet))
            img6.add(Text("DRUNet").scale(label_scale).next_to(img6, .2*UP))
            img6.height = 2.4
            imgs.append(img6)
            g = Group(*imgs).arrange().shift(2*UP)

            code = '''
            import deepinv as dinv

            ''' + operator_code + '''
            y = physics(x)

            denoiser1 = dinv.models.MedianFilter()
            denoiser2 = dinv.models.TGV()
            denoiser3 = dinv.models.BM3D()
            denoiser4 = dinv.models.DRUNet(pretrained='download', ...)
            
            ''' + f'xhat = denoiserx(y, sigma={sigma:.2f})' + '''
            '''
            rendered_code = Code(code=code, tab_width=4, background="window",
                                 language="Python", font="Monospace").shift(2 * DOWN).scale(0.7)

            tex = Tex(r"Gaussian denoising: $y\sim \mathcal{N}(x,I\sigma^2)$, $\sigma=" + f"{sigma:.2f}" + "$", font_size=32)
            self.add(tex)
            self.add(rendered_code)
            self.add(g)
            self.wait(total_slide_time/params)
            self.remove(g)
            self.remove(tex)
            self.remove(rendered_code)
