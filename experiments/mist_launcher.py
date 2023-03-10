from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def common_options(self):
        return [
            # Command 0
            Options(
                dataroot="datasets/MIST/HER2/TrainValAB",
                name="test",
                checkpoints_dir='checkpoints',
                model='cpt',
                CUT_mode="FastCUT",

                n_epochs=30,  # number of epochs with the initial learning rate
                n_epochs_decay=10,  # number of epochs to linearly decay learning rate to zero

                netD='n_layers',  # ['basic', 'n_layers, 'pixel', 'patch'], 'specify discriminator architecture. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
                ndf=32,
                netG='resnet_6blocks',  # ['resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128', 'stylegan2', 'smallstylegan2', 'resnet_cat'], 'specify generator architecture')
                n_layers_D=5,  # 'only used if netD==n_layers'
                normG='instance',  # ['instance, 'batch, 'none'], 'instance normalization or batch normalization for G')
                normD='instance',  # ['instance, 'batch, 'none'], 'instance normalization or batch normalization for D')
                weight_norm='spectral',

                lambda_GAN=1.0,  # weight for GAN loss：GAN(G(X))
                lambda_NCE=10.0,  # weight for NCE loss: NCE(G(X), X)
                nce_layers='0,4,8,12,16',
                nce_T=0.07,
                num_patches=256,

                # FDL:
                lambda_gp=10.0,
                gp_weights='[0.015625,0.03125,0.0625,0.125,0.25,1.0]',
                lambda_asp=10.0,  # weight for NCE loss: NCE(G(X), X)
                asp_loss_mode='lambda_linear',

                dataset_mode='aligned',  # chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
                direction='AtoB',
                # serial_batches='', # if true, takes images in order to make batches, otherwise takes them randomly
                num_threads=15,  # '# threads for loading data')
                batch_size=1,  # 'input batch size')
                load_size=1024,  # 'scale images to this size')
                crop_size=512,  # 'then crop to this size')
                preprocess='crop',  # ='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
                # no_flip='',
                flip_equivariance=False,
                display_winsize=512,  # display window size for both visdom and HTML
                # display_id=0,
                update_html_freq=100,
                save_epoch_freq=5,
                # print_freq=10,
            ),
        ]

    def commands(self):
        return ["python train.py " + str(opt) for opt in self.common_options()]

    def test_commands(self):
        opts = self.common_options()
        phase = 'val'
        for opt in opts:
            opt.set(crop_size=1024, num_test=1000)
            opt.remove('n_epochs', 'n_epochs_decay', 'update_html_freq',
                       'save_epoch_freq', 'continue_train', 'epoch_count')
        return ["python test.py " + str(opt.set(phase=phase)) for opt in opts]
