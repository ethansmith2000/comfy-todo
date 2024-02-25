An implementation of ToDo: Token Downsampling for Efficient Generation of High-Resolution Images https://arxiv.org/abs/2402.13573 for comfyui

Usage:
Downsample_depth_1 is how much to downsample attention at the first level. 

Downsample_depth_2 is how much to downsample attention at the second level. 


In general, I've tried to use numbers that divide evenly but it should still work otherwise

some examples:
1024x1024:
- Downsample_depth_1 2.0
- Downsample_depth_2 1.0
1536x1536:
- Downsample_depth_1 3.0
- Downsample_depth_2 1.0
2048x2048:
- Downsample_depth_1 4.0
- Downsample_depth_2 2.0

Note: for SDXL we dont actually do attention for the first depth, so that setting will be irrelevant


original repo here: https://github.com/ethansmith2000/ImprovedTokenMerge/blob/main/README.md