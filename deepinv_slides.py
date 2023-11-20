import torch
from manim import *
import sys
import os
sys.path.insert(1, '/projects/UDIP/deepinv')
import deepinv as dinv
from helper_funcs import find_operator, get_dataset, get_params, get_unrolled

def torch2np(x):
    x[x==0] = .1
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



class DeepInvOptim(Scene):
    def get_image(self, x, title, height=4., label_scale=.3):
        img = ImageMobject(torch2np(x))
        img.add(Text(title).scale(label_scale).next_to(img, .2*UP))
        img.height = height
        return img

    def my_construct(self, y, imgs, graph, code):
        fps = 5
        frame_int = max(1, int(len(imgs)/(total_slide_time*fps)))
        n_frames = int(len(imgs)/frame_int)
        print(f'frame_int: {frame_int}, n_frames: {n_frames}')

        axes = Axes(x_range=[0, len(imgs), int(len(imgs)/6)], y_range=[0, 35, 5], x_length=7, y_length=6).add_coordinates()
        axes.height = 6
        axes.width = 4
        self.add(axes)
        axes.shift(5*RIGHT).shift(1 * UP)
        title = Text("PSNR [dB]").next_to(axes, UP)
        title.scale(.6)
        self.add(title)
        listx = []
        listy = []
        k = 0
        rendered_code = Code(code=code, tab_width=4, background="window",
                             language="Python", font="Monospace").shift(2 * DOWN).scale(0.7)
        meas = self.get_image(y, title='measurement').shift(5*LEFT).shift(1 * UP)
        self.add(meas)
        self.add(rendered_code)

        for f in range(n_frames):
            x = imgs[f*frame_int]
            point = graph[f*frame_int]
            img = self.get_image(x, title='estimate').shift(1 * UP)
            self.add(img)
            listx.append(k*frame_int)
            listy.append(point)
            line = axes.plot_line_graph(listx, listy, add_vertex_dots=False, stroke_width=.5)
            self.add(line)
            self.wait(total_slide_time/n_frames)
            self.remove(img)
            self.remove(line)
            k += 1


        x = imgs[-1]
        point = graph[-1]
        img = self.get_image(x, title='estimate').shift(1 * UP)
        self.add(img)
        listx.append(len(imgs))
        listy.append(point)
        line = axes.plot_line_graph(listx, listy, add_vertex_dots=False, stroke_width=.5)
        self.add(line)
        self.wait(1)


class DeepInvSampling(Scene):
    def get_image(self, x, title, height=4., label_scale=.3):
        img = ImageMobject(torch2np(x), resampling_algorithm=RESAMPLING_ALGORITHMS["cubic"])
        img.add(Text(title).scale(label_scale).next_to(img, .2*UP))
        img.height = height
        return img

    def create_histogram(self, data, length=4, bins=16, drange=(0, 1), xlabel="PSNR [dB]"):
        # Define histogram parameters
        bin_width = length/bins

        bin_height = bin_width
        hist, bin_edges = np.histogram(data, bins=bins, range=drange)
        # Create a list of bars
        bars = VGroup()
        for i in range(len(hist)):
            count = hist[i]
            bar = Rectangle(
                height=count * bin_height,
                width=bin_width,
                fill_opacity=0.7,
            ).shift(i * bin_width * RIGHT + bin_height * count * UP / 2)
            bars.add(bar)

        # Create histogram axes
        x_axis = NumberLine(x_range=np.round([bin_edges[0], bin_edges[-1], bin_edges[2]-bin_edges[0]], 2), length=length,
                            font_size=16,
                            include_numbers=True).move_to(bars.get_bottom() + bin_height/2 * DOWN)
        #y_axis = NumberLine(x_range=[0, max(data) + 1], include_numbers=True).next_to(x_axis_label, UP)

        x_axis_label = Text(xlabel).next_to(x_axis, .5 * DOWN).scale(.4)
        # Group elements
        histogram = VGroup(x_axis, x_axis_label, bars)

        return histogram

    def my_construct(self, y, imgs, graph, code):
        fps = 5
        frame_int = max(1, int(len(imgs)/(total_slide_time*fps)))
        n_frames = int(len(imgs)/frame_int)

        rendered_code = Code(code=code, tab_width=4, background="window",
                             language="Python", font="Monospace").shift(2 * DOWN).scale(0.7)
        meas = self.get_image(y, title='measurement').shift(5*LEFT).shift(1 * UP)
        self.add(meas)
        self.add(rendered_code)

        k = 0
        for f in range(1, n_frames):
            x = imgs[f*frame_int]
            img = self.get_image(x, title='posterior sample').shift(1 * UP)
            self.add(img)
            hist = self.create_histogram(graph[:f], drange=(min(graph), max(graph))).shift(3*RIGHT)

            #title = Text("PSNR [dB]").next_to(hist, UP)
            #title.scale(.6)
            #self.add(title)
            self.add(hist)
            self.wait(total_slide_time/n_frames)
            self.remove(img)
            self.remove(hist)
            #self.remove(title)
            k += 1




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
                diffusion_model = dinv.sampling.DDRM(denoiser=drunet)
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



class DPS(DeepInvOptim):
    def construct(self):
        device = 'cuda:0'
        dataset = get_dataset("inpainting")
        x = dataset[56][0].unsqueeze(0).to(device)
        model = dinv.sampling.DPS(model=dinv.models.DiffUNet().to(device), data_fidelity=dinv.optim.L2(), verbose=True,
                                  save_iterates=True, device=device, max_iter=1000)
        physics = dinv.physics.Inpainting(tensor_size=x.shape[1:], mask=.3, device=device,
                                          noise_model=dinv.physics.GaussianNoise(sigma=.05))
        y = physics(x)
        imgs = model(y, physics)

        imgs = imgs[700:]

        code = '''
        import deepinv as dinv
        model = dinv.sampling.DPS(model=dinv.models.DiffUNet())
        xhat = model(y, physics)
        '''
        psnr = []
        for k, img in enumerate(imgs):
            psnr.append(dinv.utils.cal_psnr(img.to(device), x))

        self.my_construct(physics.A_adjoint(y), imgs, psnr, code)


class ULA(DeepInvSampling):
    def construct(self):
        device = 'cuda:0'
        dataset = get_dataset("inpainting")
        x = dataset[22][0].unsqueeze(0).to(device)

        prior = dinv.optim.ScorePrior(dinv.models.DnCNN(pretrained='download_lipschitz').to(device))
        sigma = 0.01
        sigma_den = 0.02
        burnin = .3
        MC = 100
        thinning = 30
        norm = 1.
        alpha = 20*torch.tensor(norm, device=device)

        step_size = float(1. / (norm / (sigma ** 2) + alpha / (sigma_den ** 2)))
        model = dinv.sampling.ULA(prior, data_fidelity=dinv.optim.L2(sigma), step_size=step_size,
                                  sigma=sigma_den, alpha=alpha, verbose=True,
                                  max_iter=int(MC * thinning / (.95 - burnin)),
                                  thinning=thinning, save_chain=True, burnin_ratio=burnin, clip=(-1., 2),
                                  thresh_conv=1e-4)

        #mask = torch.ones_like(x).squeeze(0)
        #mask[:, 120:140, :] = 0
        mask = .3
        physics = dinv.physics.Inpainting(tensor_size=x.shape[1:], mask=mask, device=device,
                                          noise_model=dinv.physics.GaussianNoise(sigma=sigma))
        y = physics(x)
        mean, var = model(y, physics)

        imgs = model.get_chain()

        code = '''
        import deepinv as dinv
        model = dinv.sampling.ULA(...)
        xhat = model(y, physics)
        '''
        psnr = []
        for k, img in enumerate(imgs):
            psnr.append(dinv.utils.cal_psnr(img.to(device), mean))

        self.my_construct(physics.A_adjoint(y), imgs, psnr, code)


class DiffSamples(DeepInvSampling):
    def construct(self):
        device = 'cuda:0'
        dataset = get_dataset("inpainting")
        x = dataset[22][0].unsqueeze(0).to(device)

        sigma = 0.01

        diff = dinv.sampling.DDRM(dinv.models.DRUNet().to(device))
        model = dinv.sampling.DiffusionSampler(diff, max_iter=100, save_chain=True)

        mask = .3
        physics = dinv.physics.Inpainting(tensor_size=x.shape[1:], mask=mask, device=device,
                                          noise_model=dinv.physics.GaussianNoise(sigma=sigma))
        y = physics(x)
        mean, var = model(y, physics)

        imgs = model.get_chain()

        code = '''
        import deepinv as dinv
        model = dinv.sampling.DDRM(...)
        xhat = model(y, physics)
        '''
        psnr = []
        for k, img in enumerate(imgs):
            psnr.append(dinv.utils.cal_psnr(img.to(device), mean))

        self.my_construct(physics.A_adjoint(y), imgs, psnr, code)



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
