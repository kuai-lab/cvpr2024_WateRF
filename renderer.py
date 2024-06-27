import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
from dataLoader.ray_utils import ndc_rays_blender
import torchvision.transforms as transforms
from pytorch_wavelets import DWTInverse, DWTForward
from models.attack import Attacker

# renderer
def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):

    rgbs, alphas, depth_map, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
    
        rgb_map, _ = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

        rgbs.append(rgb_map)
        # depth_maps.append(depth_map)
    
    return torch.cat(rgbs), None, None, None, None

def bit_acc(decoded, keys):
    diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
    return bit_accs

@torch.no_grad()
def evaluation(test_dataset, tensorf, msg_decoder, key, args, renderer, savePath=None, N_vis=-1, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps = [], []
    ssims, l_alex, l_vgg, bit_acc_list = [], [], [], []
    os.makedirs(savePath, exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])

        rgb_map, _, _, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                       ndc_ray=ndc_ray, white_bg=white_bg, device=device)
        rgb_map_for_decoder = rgb_map.view(H, W, 3).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map = rgb_map.reshape(H, W, 3).cpu()
        
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                
                yl, yh = DWTForward(wave='bior4.4', J=2, mode='periodization').to(rgb_map_for_decoder.device)(rgb_map_for_decoder)

                decoded = msg_decoder(yl) # b c h w -> b k
                bit_accuracy = bit_acc(decoded, key).item()
                    
                bit_acc_list.append(bit_accuracy)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        rgb_maps.append(rgb_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)

    # imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            bit_acc_ = np.mean(np.asarray(bit_acc_list))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v, bit_acc_]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

    return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset, tensorf, c2ws, renderer, savePath=None, N_vis=-1, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps = [], []
    ssims, l_alex, l_vgg = [], [], []
    os.makedirs(savePath, exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, _, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                       ndc_ray=ndc_ray, white_bg=white_bg, device=device)
        
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map = rgb_map.reshape(H, W, 3).cpu()

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        rgb_maps.append(rgb_map)
        
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

    return PSNRs

def evaluation_bit_accuracy(render_path, msg_decoder, key):
    transform_imnet = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    img_list = os.listdir(render_path)
    
    bit_acc_list = []
    
    for img_id in img_list:
        try: 
            img = Image.open(render_path + img_id)
            img = transform_imnet(img).unsqueeze(0).to("cuda")
        except:
            continue
        
        yl, yh = DWTForward(wave='bior4.4', J=2, mode='periodization').to(img.device)(img)
        decoded = msg_decoder(yl) # b c h w -> b k
        bit_accuracy = bit_acc(decoded, key).item()
        # print("Bit accuracy: ", bit_acc)
        bit_acc_list.append(bit_accuracy)
            
    return np.mean(bit_acc_list)

def evalutaion_attack_bit_accuracy(render_path, msg_decoder, key, renderer, savePath=None, device='cuda'):
    transform_imnet = transforms.Compose([
        transforms.ToTensor(),
    ])

    attack_bit_acc_result = []
    att = Attacker()
    attack_type = ['Blur','Rotate', 'Crop', 'Resize', 'noise', 'JPEG_Compression']
    img_list = os.listdir(render_path)
    
    for idx, item in enumerate(attack_type):
        total_bit_acc = 0
        total_img_num = 0
        result_dict = {}
        for img_id in img_list:
            try:
                img = Image.open(render_path + img_id)
            except:
                continue
            attacked_image = att(img,idx)
            tensored_img = transform_imnet(attacked_image).unsqueeze(0).contiguous().to(device)
            yl, yh = DWTForward(wave='bior4.4', J=2, mode='periodization').to(tensored_img.device)(tensored_img)
            decoded = msg_decoder(yl)
            total_bit_acc += bit_acc(decoded, key).item()
            total_img_num += 1
        
        result_dict['Attack_Type'] = item
        result_dict['bit_acc'] = (total_bit_acc / total_img_num)
        attack_bit_acc_result.append(result_dict)
        np.savetxt(f'{savePath}/attack_bit_acc_mean.txt',np.asarray(attack_bit_acc_result),fmt='%s')

    # return attack_bit_acc_result
