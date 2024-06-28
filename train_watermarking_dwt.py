import os
from tqdm.auto import tqdm
from opt import config_parser

import json, random
from renderer import *
from renderer import evaluation_bit_accuracy
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

from dataLoader import dataset_dict
import sys

'''Import for Watermarking '''
from loss.loss_provider import LossProvider
import math
import util.ssim as ssim_utils

import lpips

import shutil
from os import path
import torchvision.transforms as transforms

from pytorch_wavelets import DWTInverse, DWTForward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

renderer = OctreeRender_trilinear_fast

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


@torch.no_grad()
def export_mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)

def bit_acc(decoded, keys):
    diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
    return bit_accs

lpips_alex = lpips.LPIPS(net='alex') # best forward scores
lpips_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

def ssim(img1, img2, window_size = 11, size_average = True, format='NCHW'):
    if format == 'HWC':
        img1 = img1.permute([2, 0, 1])[None, ...]
        img2 = img2.permute([2, 0, 1])[None, ...]
    elif format == 'NHWC':
        img1 = img1.permute([0, 3, 1, 2])
        img2 = img2.permute([0, 3, 1, 2])

    return ssim_utils.ssim(img1, img2, window_size, size_average)

def lpips(img1, img2, net='alex', format='NCHW'):

    if format == 'HWC':
        img1 = img1.permute([2, 0, 1])[None, ...]
        img2 = img2.permute([2, 0, 1])[None, ...]
    elif format == 'NHWC':
        img1 = img1.permute([0, 3, 1, 2])
        img2 = img2.permute([0, 3, 1, 2])

    if net == 'alex':
        return lpips_alex(img1, img2)
    elif net == 'vgg':
        return lpips_vgg(img1, img2)

def total_variation_loss(image):

    batch_size, num_channels, height, width = image.size()

    horizontal_grad = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
    vertical_grad = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])

    tv_loss = torch.mean(horizontal_grad) + torch.mean(vertical_grad)

    return tv_loss

def finetuning_deferred_backpropagation(args):

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'

    # make log folder
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/model_weight', exist_ok=True)
    
    # save code and parameters
    with open(path.join(logfolder, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        # changed name to prevent errors
        shutil.copyfile(__file__, path.join(logfolder, 'train_frozen.py'))
    
    global_start_time = datetime.datetime.now()
    
    '''Load models & keys & losses'''
    
    print(f"/n>>> Loading model from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    
    # Load decoder for watermarking
    print(f'\n>>> Loading decoder from {args.msg_decoder_path}...')
    
    msg_decoder = torch.jit.load(args.msg_decoder_path).to(device)

    msg_decoder.eval()
    nbit = msg_decoder(torch.zeros(1, 3, 128, 128).to(device)).shape[-1]
    
    # Creating key
    print(f'\n>>> Creating key with {nbit} bits...')
    
    '''random key'''
    key = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device=device)
    key_str = "".join([ str(int(ii)) for ii in key.tolist()[0]])

    print(f'Key: {key_str}')
    key_file = open(os.path.join(logfolder, "key.txt"), "w")
    key_file.write(key_str)
    key_file.close()
    
    # Define Loss
    print(f'\n>>> Creating losses...')
    print(f'Losses: {args.loss_w} and {args.loss_i}...')

    # Message Loss
    if args.loss_w == 'mse':        
        loss_w = lambda decoded, keys, temp=10.0: torch.mean((decoded*temp - (2*keys-1))**2) # b k - b k
    elif args.loss_w == 'bce':
        loss_w = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(decoded*temp, keys, reduction='mean')
    else:
        raise NotImplementedError

    # Image Loss
    if args.loss_i == 'mse':
        loss_i = lambda imgs_w, imgs: torch.mean((imgs_w - imgs)**2)
    elif args.loss_i == 'watson-dft':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    elif args.loss_i == 'watson-vgg':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    elif args.loss_i == 'ssim':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('SSIM', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    else:
        raise NotImplementedError
    
    # Image Loss (RGB) in deferred-backpropagation (patch-wise)
    loss_rgb = nn.L1Loss(reduction='mean').to(device)
    
    provider_deferred = LossProvider()
    loss_percep_deferred = provider_deferred.get_loss_function('SSIM', colorspace='RGB', pretrained=True, reduction='sum')
    loss_percep_deferred = loss_percep_deferred.to(device)
    loss_perceptual = lambda imgs_w, imgs: loss_percep_deferred((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis) 
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))
    
    summary_writer = SummaryWriter(logfolder)
    
    # Load dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    # Handling different dataset
    try: # llff dataset
        train_images_num = train_dataset.num_images
    except: # synthetic dataset
        train_images_num = len(train_dataset.image_paths)
    image_w, image_h = train_dataset.img_wh
    torch.cuda.empty_cache()
    
    # Set trainabe lambda_i
    initial_lambda_i = args.lambda_i

    lambda_i = nn.Parameter(torch.tensor(initial_lambda_i))
    optimizer_lambda = torch.optim.Adam([{'params': lambda_i, "lr" : 0.0003}], betas=(0.9,0.99))

    # Train with deferred-backpropagation
    gstep = 0
    pbar = tqdm(range(0, args.epoch), total = args.epoch)
    image_list = list(range(train_images_num))
    
    lambda_i_list = []
    for epoch_id in pbar:
        if epoch_id ==0:
            lambda_i.requires_grad_(False)
            
        if epoch_id ==4:
            lambda_i.requires_grad_(True)
        stats = {}
        random.shuffle(image_list)
        
        for img_i in image_list:
            gstep+=1
            
            rays_train = train_dataset.all_rays.view(train_images_num, image_h, image_w, 6)[img_i].view(-1, 6)
            rgb_train = train_dataset.all_rgbs.view(train_images_num, image_h, image_w, 3)[img_i].view(-1, 3)

            with torch.no_grad():
                rgb_pred_wm, _, _, _, _ = renderer(rays_train, tensorf, chunk=args.batch_size,
                                        N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device, is_train=False)

                rgb_pred_wm = rgb_pred_wm.view(image_h, image_w, 3).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
                rgb_gt = rgb_train.view(image_h, image_w, 3).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
            

            rgb_pred_wm.requires_grad_(True)
            rgb_pred_wm_im = rgb_pred_wm.clone().detach()
            
            #DWT transform
            yl, yh = DWTForward(wave=args.dwt_wave, J=args.dwt_level, mode=args.dwt_mode).to(rgb_pred_wm.device)(rgb_pred_wm)
            
            # Extract watermark
            decoded = msg_decoder(yl) # b c h w -> b k
            
            loss_wm = loss_w(decoded, key)
            loss_im = loss_i(rgb_pred_wm, rgb_gt)
            loss_im_mse = F.mse_loss(rgb_gt, rgb_pred_wm) # Only for psnr
            psnr = -10.0 * math.log10(loss_im_mse)
            
            
            # Total loss
            total_loss = lambda_i * loss_im + (1-lambda_i) * loss_wm
            total_loss = total_loss.mean()
            
            # No need to mean() because of batch size 1
            loss_dict = {}
            loss_dict['watermark_loss'] = loss_wm.detach().item()
            loss_dict['image_loss'] = loss_im.detach().item()
            loss_dict['psnr'] = psnr
            loss_dict['ssim'] = ssim(rgb_pred_wm.squeeze(0).permute(1,2,0), rgb_gt.squeeze(0).permute(1,2,0), format='HWC').item()
            loss_dict['lpips'] = lpips(rgb_pred_wm.squeeze(0).permute(1,2,0).cpu(), rgb_gt.squeeze(0).permute(1,2,0).cpu(), format='HWC').item()
            loss_dict['bit-accuracy'] = bit_acc(decoded, key).item()
            loss_dict['total_loss'] = total_loss.item()
                
            optimizer_lambda.zero_grad()
            total_loss.backward()

            if epoch_id >= 4:
                lambda_i.grad *= -1 
            
            optimizer_lambda.step()

            # keep lambda to 0.1 ~ 0.9
            lambda_i.data = torch.clamp(lambda_i, min=0.1, max=0.9)

            lambda_i_list.append(float(lambda_i.detach().numpy()))

            rgb_pred_wm_grad = rgb_pred_wm.grad.squeeze(0).permute(1, 2, 0).contiguous().clone().detach().view(-1, 3)

            optimizer.zero_grad()
            

            '''deferred-backpropagation'''
            for batch_start in range(0, image_h*image_w, args.batch_size):
                rgb_pred_wm, _, _, _, _ = renderer(rays_train[batch_start:batch_start+args.batch_size], tensorf, chunk=args.batch_size,
                                        N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)
                rgb_pred_wm.requires_grad_(True)
                
                rgb_pred_wm.backward(rgb_pred_wm_grad[batch_start:batch_start+args.batch_size], retain_graph=True) 
                rgb_patch_gt = rgb_train[batch_start:batch_start+args.batch_size].to(rgb_pred_wm.device)
                
                # rgb_loss = 0.1 * loss_rgb(rgb_pred_wm, rgb_patch_gt)
                rgb_loss =  loss_rgb(rgb_pred_wm, rgb_patch_gt)
                
                patch_size = int(math.sqrt(rgb_pred_wm.shape[0]))

                try:
                    rgb_pred_patch = rgb_pred_wm.view(patch_size, patch_size, 3).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
                    rgb_gt_patch = rgb_patch_gt.view(patch_size, patch_size, 3).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
                    ssim_loss = loss_perceptual(rgb_pred_patch, rgb_gt_patch)
                    tv_loss = total_variation_loss(rgb_pred_patch)
                except:
                    ssim_loss = 0
                    tv_loss = 0

                patch_loss = args.lambda_l1*rgb_loss + args.lambda_ssim*ssim_loss + args.lambda_tv*tv_loss
                patch_loss.backward()

            loss_dict['patch_loss'] = patch_loss.item()

            optimizer.step()
            
            torch.cuda.empty_cache()
            
            # Select IDs FOR VIEWING
            if img_i in [0,1,2]:
                # image
                summary_writer.add_image(f'train_image/gt_{img_i:04d}', rgb_gt.squeeze(0).permute(1,2,0).cpu().clamp_max_(1.0), global_step=gstep, dataformats='HWC')
                summary_writer.add_image(f'train_image/pred_wm_{img_i:04d}', rgb_pred_wm_im.squeeze(0).permute(1,2,0).cpu().clamp_max_(1.0), global_step=gstep, dataformats='HWC')
                # DWT
                summary_writer.add_image(f'train_image/LL_{img_i:04d}', yl.squeeze(0).permute(1,2,0).cpu().clamp_max_(1.0), global_step=gstep, dataformats='HWC')
            # Stats
            for x in loss_dict:
                stats[x] = loss_dict[x]

            for stat_name in stats:
                stat_train = stats[stat_name] # / args.print_every
                summary_writer.add_scalar('train/' + stat_name, stat_train, global_step=gstep)

            print()
            # pbar.set_description(f'global-iteration-{gstep} | epoch-{epoch_id+1}, IMG_ID-{img_i} : loss_im={loss_dict["image_loss"]:.4f} loss_wm={loss_dict["watermark_loss"]:.4f} bit-accuracy={loss_dict["bit-accuracy"]:.4f} psnr={loss_dict["psnr"]:.4f} ssim={loss_dict["ssim"]:.4f} lpips={loss_dict["lpips"]:.4f} ')
            pbar.set_description(f'global-iteration-{gstep} | epoch-{epoch_id+1} : loss_im={loss_dict["image_loss"]:.4f} loss_wm={loss_dict["watermark_loss"]:.4f} bit-accuracy={loss_dict["bit-accuracy"]:.4f} psnr={loss_dict["psnr"]:.4f} ssim={loss_dict["ssim"]:.4f} lpips={loss_dict["lpips"]:.4f} ')

            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_factor
                
        # Validation
        loss_dict_test = {"watermark_loss":0.0, "image_loss":0.0, "psnr":0.0, "ssim":0.0, "lpips":0.0, "bit-accuracy":0.0, "total_loss":0.0, "GT bit-accuracy (It must be low)": 0.0}
        
        try: # llff dataset
            test_images_num = test_dataset.num_images
        except: # synthetic dataset
            test_images_num = len(test_dataset.image_paths)
            
        val_num = args.val_num
        
        print(f'\nValidation')
        
        for img_i in range(val_num):
            
            with torch.no_grad():
                rays_test = test_dataset.all_rays.view(test_images_num, image_h, image_w, 6)[img_i].view(-1, 6)
                rgb_test = test_dataset.all_rgbs.view(test_images_num, image_h, image_w, 3)[img_i].view(-1, 3)
                
                rgb_pred_wm_test, _, _, _, _ = renderer(rays_test, tensorf, chunk=args.batch_size,
                                        N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device, is_train=False)

                rgb_pred_wm_test = rgb_pred_wm_test.view(image_h, image_w, 3).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
                rgb_gt_test = rgb_test.view(image_h, image_w, 3).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
                
                yl, yh = DWTForward(wave=args.dwt_wave, J=args.dwt_level, mode=args.dwt_mode).to(rgb_pred_wm_test.device)(rgb_pred_wm_test)
                # just for test; no watermarked dwt (GT)
                no_wm_yl, no_wm_yh = DWTForward(wave=args.dwt_wave, J=args.dwt_level, mode=args.dwt_mode).to(rgb_gt_test.device)(rgb_gt_test)

                # Extract watermark
                decoded = msg_decoder(yl) # b c h w -> b k
                
                loss_wm = loss_w(decoded, key)
                loss_im = loss_i(rgb_pred_wm_test, rgb_gt_test)
                loss_im_mse = F.mse_loss(rgb_gt_test, rgb_pred_wm_test) # Only for psnr
                psnr = -10.0 * math.log10(loss_im_mse)
                
                # Total loss
                total_loss = lambda_i * loss_im + (1-lambda_i) * loss_wm
                total_loss = total_loss.mean()
                
                # It needs to mean() later
                loss_dict_test['watermark_loss'] += loss_wm.detach().item()
                loss_dict_test['image_loss'] += loss_im.detach().item()
                loss_dict_test['psnr'] += psnr
                loss_dict_test['ssim'] += ssim(rgb_pred_wm_test.squeeze(0).permute(1,2,0), rgb_gt_test.squeeze(0).permute(1,2,0), format='HWC').item()
                loss_dict_test['lpips'] += lpips(rgb_pred_wm_test.squeeze(0).permute(1,2,0).cpu(), rgb_gt_test.squeeze(0).permute(1,2,0).cpu(), format='HWC').item()
                loss_dict_test['bit-accuracy'] += bit_acc(decoded, key).item()
                loss_dict_test['total_loss'] += total_loss.item()
                # GT bit-accuracy test
                loss_dict_test["GT bit-accuracy (It must be low)"] += bit_acc(msg_decoder(no_wm_yl), key).item()
                
            if img_i in [0,1,2]:
                summary_writer.add_image(f'test_image/gt_{img_i:04d}', rgb_gt_test.squeeze(0).permute(1,2,0).cpu().clamp_max_(1.0), global_step=gstep, dataformats='HWC')
                summary_writer.add_image(f'test_image/pred_wm_{img_i:04d}', rgb_pred_wm_test.squeeze(0).permute(1,2,0).cpu().clamp_max_(1.0), global_step=gstep, dataformats='HWC')                
                summary_writer.add_image(f'test_image/dwt_{img_i:04d}', yl.squeeze(0).permute(1,2,0).cpu().clamp_max_(1.0), global_step=gstep, dataformats='HWC')

        for stat_name in loss_dict_test:
            loss_dict_test[stat_name] /= val_num
            summary_writer.add_scalar('test/' + stat_name, loss_dict_test[stat_name], global_step=gstep)
        
        print('<validation stats>\n')
        for stat_name in loss_dict_test:
            print(f'{stat_name} : {loss_dict_test[stat_name]:.4f}')
        print()

    # save model
    print(f'\n>>> Save model in {logfolder}/{args.expname}.th ...')
    tensorf.save(f'{logfolder}/model_weight/{args.expname}.th')

    if args.render_train:
        os.makedirs(f'{logfolder}/rendering_imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, msg_decoder, key, args, renderer, f'{logfolder}/rendering_imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/rendering_imgs_test_all_{args.expname}', exist_ok=True)
        PSNRs_test = evaluation(test_dataset,tensorf, msg_decoder, key, args, renderer, f'{logfolder}/rendering_imgs_test_all_{args.expname}/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        evalutaion_attack_bit_accuracy(f'{logfolder}/rendering_imgs_test_all_{args.expname}/',msg_decoder,key,renderer, f'{logfolder}/rendering_imgs_test_all_{args.expname}/', device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        savePath = f'{logfolder}/rendering_imgs_path_all_{args.expname}/'
        os.makedirs(f'{logfolder}/rendering_imgs_path_all_{args.expname}/', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/rendering_imgs_path_all_{args.expname}/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        bit_acc_ = evaluation_bit_accuracy(f'{logfolder}/rendering_imgs_path_all_{args.expname}/', msg_decoder, key)
        np.savetxt(f'{savePath}/bit_acc_mean.txt', np.asarray([bit_acc_]))

        evalutaion_attack_bit_accuracy(f'{logfolder}/rendering_imgs_path_all_{args.expname}/',msg_decoder,key,renderer, f'{logfolder}/rendering_imgs_path_all_{args.expname}/', device=device)

    # save lambda_i_list
    lambda_path = logfolder + "/lambda_i_list.txt"

    with open(lambda_path, "w") as file:
        for item in lambda_i_list:
            file.write(str(item) + "\n")        
        
    global_stop_time = datetime.datetime.now()
    secs = (global_stop_time - global_start_time).total_seconds()
    timings_file = open(os.path.join(logfolder, "time_mins.txt"), "w")
    timings_file.write(f"{secs / 60}\n")
    timings_file.close()
    
@torch.no_grad()
def render_test(args):
    logfolder = os.path.dirname(os.path.dirname(args.ckpt))
    
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
    
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    # Load decoder for watermarking
    print(f'\n>>> Loading decoder from {args.msg_decoder_path}...')
    msg_decoder = torch.jit.load(args.msg_decoder_path).to(device)

    msg_decoder.eval()
    
    with open(f'{logfolder}/key.txt', 'r') as file:
        key = file.read().strip()
        key = torch.tensor([int(ii) for ii in str(key)], dtype=torch.float32, device=device)


    if args.render_train:
        os.makedirs(f'{logfolder}/rendering_imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, msg_decoder, key, args, renderer, f'{logfolder}/rendering_imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/rendering_imgs_test_all_{args.expname}', exist_ok=True)
        PSNRs_test = evaluation(test_dataset,tensorf, msg_decoder, key, args, renderer, f'{logfolder}/rendering_imgs_test_all_{args.expname}/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        evalutaion_attack_bit_accuracy(f'{logfolder}/rendering_imgs_test_all_{args.expname}/',msg_decoder,key,renderer, f'{logfolder}/rendering_imgs_test_all_{args.expname}/', device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        savePath = f'{logfolder}/rendering_imgs_path_all_{args.expname}/'
        os.makedirs(f'{logfolder}/rendering_imgs_path_all_{args.expname}/', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/rendering_imgs_path_all_{args.expname}/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        bit_acc_ = evaluation_bit_accuracy(f'{logfolder}/rendering_imgs_path_all_{args.expname}/', msg_decoder, key)
        np.savetxt(f'{savePath}/bit_acc_mean.txt', np.asarray([bit_acc_]))

        evalutaion_attack_bit_accuracy(f'{logfolder}/rendering_imgs_path_all_{args.expname}/',msg_decoder,key,renderer, f'{logfolder}/rendering_imgs_path_all_{args.expname}/', device=device)


def reconstruction(args):

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray 

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)



    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))


    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct)


    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))


    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]


    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:


        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        #rgb_map, alphas_map, depth_map, weights, uncertainty
        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, tensorf, chunk=args.batch_size,
                                N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)
        
        
        loss = torch.mean((rgb_map - rgb_train) ** 2)


        # loss
        total_loss = loss
        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight*loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight*loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if TV_weight_density>0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app>0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()
        
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)


        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []


        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)



        if iteration in update_AlphaMask_list:

            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)


            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays,allrgbs = tensorf.filtering_rays(allrays,allrgbs)
                trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)


        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        

    tensorf.save(f'{logfolder}/{args.expname}.th')


    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    # torch.manual_seed(19701105)
    # np.random.seed(19701105)
    # random_seed = random.randint(0, 100000)
    # torch.manual_seed(random_seed)
    # np.random.seed(random_seed)
    seed_everything(19701105)
    args = config_parser()
    
    if  args.export_mesh:
        export_mesh(args)

    if args.render_only:
        render_test(args)
    else:
        # reconstruction(args)
        finetuning_deferred_backpropagation(args)