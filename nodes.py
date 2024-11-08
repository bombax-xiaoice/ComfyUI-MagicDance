import os
import torch
import folder_paths
import torchvision.transforms
from comfy.utils import ProgressBar, calculate_parameters, weight_dtype
from comfy.cli_args import args
from comfy import model_management
import latent_preview
import comfy.latent_formats

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

import sys
comfy_path = os.path.dirname(folder_paths.__file__)
sys.path.append(f'{comfy_path}/custom_nodes/ComfyUI-MagicDance')
print(sys.path)

from model_lib.ControlNet.cldm.model import create_model

script_directory = os.path.dirname(os.path.abspath(__file__))

class LoadMagicDanceModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config": ("STRING", {"default":"model_lib/ControlNet/models/cldm_v15_reference_only_pose.yaml"},),
                "model": ("STRING", {"default": "pretrained_weights/model_state-110000.th"}),
            }
        }
    CATEGORY = "MagicDance"
    RETURN_TYPES = ("MAGICDANCEPIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "run"
    
    def run(self, config, model):
        configpath = os.path.join(script_directory, config)
        if not os.path.exists(configpath):
            raise ValueError("model yaml file not exist, make sure to 'git clone https://github.com/Boese0601/MagicDance .' under the same directory")
        if os.path.exists(model):
            modelpath = model
        else:
            modelpath = os.path.join(script_directory, model)
            if not os.path.exists(modelpath):
                modelpath = os.path.join(comfy_path, model)
                if not os.path.exists(modelpath):
                    modelpath = os.path.join(script_directory, "pretrained_weights/model_state-110000.th")
                    if not os.path.exists(modelpath):
                        from huggingface_hub import snapshot_download
                        snapshot_download("Boese0601/MagicDance", allow_patterns=["model_state-110000.th",], local_dir=os.path.join(script_directory, "pretrained_weights"), local_dir_use_symlinks=False)
        pbar = ProgressBar(3)
        model = create_model(configpath)
        pbar.update(1)
        model.sd_locked = True
        model.only_mid_control = False
        try:
            parameters = calculate_parameters(model.state_dict())
            dtype = weight_dtype(model.state_dict())
            initdevice = model_management.unet_inital_load_device(parameters, dtype)
            state_dict = torch.load(modelpath, map_location=model_management.get_torch_device_name(initdevice))
        except:
            state_dict = torch.load(modelpath, map_location="cpu")
        pbar.update(1)
        state_dict = state_dict.get('state_dict', state_dict)
        if state_dict and 'cond_stage_model.transformer.text_model.embeddings.position_ids' in state_dict:
            del state_dict['cond_stage_model.transformer.text_model.embeddings.position_ids']
        model.load_state_dict(state_dict, strict=True)
        pbar.update(1)
        del state_dict

        return (model,)

class MagicDanceEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("MAGICDANCEPIPE",),
                "image": ("IMAGE",),
            }
        }
    CATEGORY = "MagicDance"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("reference",)
    FUNCTION = "run"
    
    def run(self, pipe, image):
        imagedevice = image.device
        imagedtype = image.dtype
        image = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image[0].permute(2, 0, 1)).unsqueeze(0)
        #sd = pipe.state_dict()
        #parameters = calculate_parameters(sd, 'first_stage_model.encoder.') + calculate_parameters(sd, 'first_stage_model.quant_conv.')
        olddtype = pipe.first_stage_model.dtype
        device = model_management.vae_device()
        dtype = model_management.vae_dtype(device)
        if pipe.first_stage_model.device != device or pipe.first_stage_model.dtype != dtype:
            try:
                pipe.first_stage_model.encoder = pipe.first_stage_model.encoder.to(device = device, dtype = dtype)
                pipe.first_stage_model.quant_conv = pipe.first_stage_model.quant_conv.to(device = device, dtype = dtype)
            except:
                model_management.unload_all_models()
                model_management.soft_empty_cache()
                pipe.first_stage_model.encoder = pipe.first_stage_model.encoder.to(device = device, dtype = dtype)
                pipe.first_stage_model.quant_conv = pipe.first_stage_model.quant_conv.to(device = device, dtype = dtype)
        if image.device != device or image.dtype != dtype:
            image = model_management.cast_to_device(image, device, dtype)
        
        reference = pipe.get_first_stage_encoding(pipe.encode_first_stage(image))

        if device != model_management.vae_offload_device() or dtype != olddtype:
            pipe.first_stage_model.encoder = pipe.first_stage_model.encoder.to(device=model_management.vae_offload_device(), dtype=olddtype)
            pipe.first_stage_model.quant_conv = pipe.first_stage_model.quant_conv.to(device=model_management.vae_offload_device(), dtype=olddtype)

        if image.device != imagedevice or image.dtype != imagedtype:
            image = model_management.cast_to_device(image, imagedevice, imagedtype)
        if reference.device != imagedevice or reference.dtype != imagedtype:
            reference = model_management.cast_to_device(reference, imagedevice, imagedtype)
        
        return ({'samples':reference},)

class MagicDanceSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("MAGICDANCEPIPE",),
                "reference": ("LATENT",),
                "poses": ("IMAGE",),
                "latents": ("LATENT",),
                "steps": ("INT", {"default":50}),
                "guidance": ("FLOAT", {"default":7.0}),
            }
        }
    CATEGORY = "MagicDance"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "run"

    def run(self, pipe, reference, poses, latents, steps, guidance):
        posesdevice = poses.device
        posesdtype = poses.dtype
        latentsdevice = latents["samples"].device
        latentsdtype = latents["samples"].dtype
        #image = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image[0].permute(2, 0, 1)).unsqueeze(0)
        #cond_image = pipe.get_first_stage_encoding(pipe.encode_first_stage(image))
        sd = pipe.state_dict()
        #parameters = calculate_parameters(sd, 'cond_stage_model.')
        olddtype = pipe.cond_stage_model.transformer.dtype
        device = model_management.text_encoder_device()
        dtype = model_management.text_encoder_dtype(device)
        if pipe.cond_stage_model.transformer.device != device or pipe.cond_stage_model.transformer.dtype != dtype:
            try:
                pipe.cond_stage_model = pipe.cond_stage_model.to(device = device, dtype = dtype)
            except:
                model_management.unload_all_models()
                model_management.soft_empty_cache()
                pipe.cond_stage_model = pipe.cond_stage_model.to(device = device, dtype = dtype)

        c_cross = pipe.get_learned_conditioning([''])[:1]
        uc_cross = pipe.get_unconditional_conditioning(1)
        
        if device != model_management.text_encoder_offload_device() or dtype != olddtype:
            pipe.cond_stage_model = pipe.cond_stage_model.to(device = model_management.text_encoder_offload_device(), dtype = olddtype)
        
        if torch.nonzero(latents["samples"]).shape[0]==0:
            latents["samples"] = torch.randn_like(latents["samples"])
        
        device = model_management.get_torch_device()
        # parameters = calculate_parameters(sd, 'model.') + calculate_parameters(sd, 'pose_control_model.') + calculate_parameters(sd, 'appearance_control_model.')
        dtype = torch.float32 #model_management.unet_dtype(device, parameters)
        olddtype = pipe.dtype
        if pipe.device != device or pipe.dtype != dtype:
            try:
                pipe.model = model_management.cast_to_device(pipe.model, device, dtype)
                pipe.pose_control_model = model_management.cast_to_device(pipe.pose_control_model, device, dtype)
                pipe.appearance_control_model = model_management.cast_to_device(pipe.appearance_control_model, device, dtype)
                pipe.betas = model_management.cast_to_device(pipe.betas, device, dtype)
            except:
                model_management.unload_all_models()
                model_management.soft_empty_cache()
                pipe.model = model_management.cast_to_device(pipe.model, device, dtype)
                pipe.pose_control_model = model_management.cast_to_device(pipe.pose_control_model, device, dtype)
                pipe.appearance_control_model = model_management.cast_to_device(pipe.appearance_control_model, device, dtype)
                pipe.betas = model_management.cast_to_device(pipe.betas, device, dtype)
        
        if c_cross.device != device or c_cross.dtype != dtype:
            c_cross = model_management.cast_to_device(c_cross, device, dtype)
        if uc_cross.device != device or uc_cross.dtype != dtype:
            uc_cross = model_management.cast_to_device(uc_cross, device, dtype)
        referencedevice = reference["samples"].device
        referencedtype = reference["samples"].dtype
        if reference["samples"].device != device or reference["samples"].dtype != dtype:
            reference["samples"] = model_management.cast_to_device(reference["samples"], device, dtype)
        gene_latent_list = []
        try:
            setattr(pipe, 'load_device', pipe.model.device)
            setattr(pipe.model, 'latent_format', comfy.latent_formats.SD15())
            callback = latent_preview.prepare_callback(pipe, poses.shape[0] * steps)
        except:
            callback = None
        for i in range(poses.shape[0]):
            pose = poses[i:i+1,:,:,:].permute(0, 3, 1, 2)
            latent = latents["samples"][i:i+1,:,:,:]
            if pose.device != device or pose.dtype != dtype:
                pose = model_management.cast_to_device(pose, device, dtype)
            if latent.device != device or latent.dtype != dtype:
                latent = model_management.cast_to_device(latent, device, dtype)
            c = {"c_concat": [pose], "c_crossattn": [c_cross], "image_control":[reference["samples"]], "wonoise": True, "overlap_sampling": False}
            uc = {"c_concat": [pose], "c_crossattn": [uc_cross], "wonoise": True, "overlap_sampling": False}
            gene_latent, _ = pipe.sample_log(cond=c,
                batch_size=1, ddim=True,
                ddim_steps=steps, eta=0.0,
                unconditional_guidance_scale=guidance,
                unconditional_conditioning=uc,
                inpaint=None,
                x_T=latent,
                img_callback = lambda x0,j:callback(i*steps+j, x0, None, poses.shape[0]*steps) if callback else None,
            )
            if pose.device != posesdevice or pose.dtype != posesdtype:
                pose = model_management.cast_to_device(pose, posesdevice, posesdtype)
            if latent.device != latentsdevice or latent.dtype != latentsdtype:
                latent = model_management.cast_to_device(pose, latentsdevice, latentsdtype)
            if gene_latent.device != latentsdevice or gene_latent.dtype != latentsdtype:
                gene_latent = model_management.cast_to_device(gene_latent, latentsdevice, latentsdtype)
            gene_latent_list.append(gene_latent)
        
        if pipe.device != model_management.unet_offload_device() or pipe.dtype != olddtype:
            pipe = model_management.cast_to_device(pipe, model_management.unet_offload_device(), olddtype)
        if pipe.model.device != model_management.unet_offload_device() or pipe.model.dtype != olddtype:
            pipe.model = model_management.cast_to_device(pipe.model, model_management.unet_offload_device(), olddtype)
            pipe.pose_control_model = model_management.cast_to_device(pipe.pose_control_model, model_management.unet_offload_device(), olddtype)
            pipe.appearance_control_model = model_management.cast_to_device(pipe.appearance_control_model, model_management.unet_offload_device(), olddtype)
            pipe.betas = model_management.cast_to_device(pipe.betas, model_management.unet_offload_device(), olddtype)
        if reference["samples"].device != referencedevice or reference["samples"].dtype != referencedtype:
            reference["samples"] = model_management.cast_to_device(reference["samples"], referencedevice, referencedtype)
        gene_latents = torch.cat(gene_latent_list, dim=0)        
        return ({"samples":gene_latents},)
    
class MagicDanceDecoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("MAGICDANCEPIPE",),
                "latents": ("LATENT",),
            }
        }
    CATEGORY = "MagicDance"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"

    def run(self, pipe, latents):
        latentsdevice = latents["samples"].device
        latentsdtype = latents["samples"].dtype
        #sd = pipe.state_dict()
        #parameters = calculate_parameters(sd, 'first_stage_model.decoder.') + calculate_parameters(sd, 'first_stage_model.post_quant_conv.')
        olddtype = pipe.first_stage_model.dtype
        device = model_management.vae_device()
        dtype = model_management.vae_dtype(device)
        if pipe.first_stage_model.device != device or pipe.first_stage_model.dtype != dtype:
            try:
                pipe.first_stage_model.decoder = pipe.first_stage_model.decoder.to(device = device, dtype = dtype)                
                pipe.first_stage_model.post_quant_conv = pipe.first_stage_model.post_quant_conv.to(device = device, dtype = dtype)
            except:
                model_management.unload_all_models()
                model_management.soft_empty_cache()
                pipe.first_stage_model.decoder = pipe.first_stage_model.decoder.to(device = device, dtype = dtype)                
                pipe.first_stage_model.post_quant_conv = pipe.first_stage_model.post_quant_conv.to(device = device, dtype = dtype)
        gene_img_list = []
        pbar = ProgressBar(latents["samples"].shape[0])
        for i in range(latents["samples"].shape[0]):
            latent = latents["samples"][i:i+1,:,:,:]
            if latent.device != device or latent.dtype != dtype:
                latent = model_management.cast_to_device(latent, device, dtype)
                
            gene_img = pipe.decode_first_stage(latent)
            
            if latent.device != latentsdevice or latent.dtype != latentsdtype:
                latent = model_management.cast_to_device(latent, latentsdevice, latentsdtype)
            if gene_img.device != latentsdevice or gene_img.dtype != latentsdtype:
                gene_img = model_management.cast_to_device(gene_img, latentsdevice, latentsdtype)
            
            gene_img_list.append(gene_img.float().clamp(-1,1).cpu().add(1).mul(0.5))
            if args.preview_method != latent_preview.LatentPreviewMethod.NoPreviews:
                pbar.update_absolute(i+1, preview=("JPEG", latent_preview.preview_to_image(gene_img[0].permute(1, 2, 0)), args.preview_size))
            else:
                pbar.update(1)

        if device != model_management.vae_offload_device() or dtype != olddtype:
            pipe.first_stage_model.decoder = pipe.first_stage_model.decoder.to(device = model_management.vae_offload_device(), dtype = olddtype)
            pipe.first_stage_model.post_quant_conv = pipe.first_stage_model.post_quant_conv.to(device = model_management.vae_offload_device(), dtype = olddtype)
        
        gene_images = torch.cat(gene_img_list, dim=0).permute(0,2,3,1)
        return (gene_images,)

NODE_CLASS_MAPPINGS = {
    "LoadMagicDanceModel":LoadMagicDanceModel,
    "MagicDanceSampler":MagicDanceSampler,
    "MagicDanceDecoder":MagicDanceDecoder,
    "MagicDanceEncoder":MagicDanceEncoder,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadMagicDanceModel":"(Down)Load MagicDance Model",
    "MagicDanceSampler":"Magic Dance Sampler",
    "MagicDanceDecoder":"Magic Dance Decoder",
    "MagicDanceEncoder":"Magic Dance Encoder",
}
