# ComfyUI-MagicDance
ComfyUI supports over Boese0601/MagicDance

# Install
pip install -r requirements.txt

You can download the model file from huggingface or its mirror site beforehand, or just wait for the first run of (Down)Load MagicDance Model to download it
wget https://huggingface.co/Boese0601/MagicDance/resolve/main/model_state-110000.th
wget https://hf-mirror.com/Boese0601/MagicDance/resolve/main/model_state-110000.th

# Example Workflow
example_data/magicdance-comfy-example.json

# Tips
Allow multiple poses (pose images) but only one single reference (image encoded to latent). The input latents should set its first dimension the same as the number of poses.

Verified to work on a single NVidia RTX 3070 card with 8G graphics memory, where VAE encoder, TextEncoder, Transformer and VAE decoder are loaded seperately. If you have enough graphics memory. You can try use --highvram on comfy start, where the entire pipeline is loaded into GPU directly to spare unnecessary conversion between CPU and GPU.

It is recommend to choose a preview method, so that you can see the progress of each pose and each sampler step during the long run.

Email zhouli@xiaobing.ai for further question or discussion.