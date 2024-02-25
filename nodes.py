import math
import torch.nn.functional as F


def up_or_downsample(item, cur_w, cur_h, new_w, new_h, method="nearest"):
    batch_size = item.shape[0]

    item = item.reshape(batch_size, cur_h, cur_w, -1)
    item = item.permute(0, 3, 1, 2)
    df = cur_h // new_h
    if method in "max_pool":
        item = F.max_pool2d(item, kernel_size=df, stride=df, padding=0)
    elif method in "avg_pool":
        item = F.avg_pool2d(item, kernel_size=df, stride=df, padding=0)
    else:
        item = F.interpolate(item, size=(new_h, new_w), mode=method)
    item = item.permute(0, 2, 3, 1)
    item = item.reshape(batch_size, new_h * new_w, -1)

    return item

def get_functions(x, downsample_factor_1, downsample_factor_2, original_shape):
    b, c, original_h, original_w = original_shape
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))
    cur_h = original_h // downsample
    cur_w = original_w // downsample

    merge_op = lambda x: x
    if downsample == 1 and downsample_factor_1 > 1:
        new_h = int(cur_h / downsample_factor_1)
        new_w = int(cur_w / downsample_factor_1)
        merge_op = lambda x: up_or_downsample(x, cur_w, cur_h, new_w, new_h)
    elif downsample == 2 and downsample_factor_2 > 1:
        new_h = int(cur_h / downsample_factor_2)
        new_w = int(cur_w / downsample_factor_2)
        merge_op = lambda x: up_or_downsample(x, cur_w, cur_h, new_w, new_h)

    return merge_op


class ToDoPatchModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "downsample_factor_depth_1": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.01}),
                                "downsample_factor_depth_2": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.01}),
                             }}

    # timestep_threshold_stop

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, downsample_factor_depth_1, downsample_factor_depth_2):
        def todo_m(q, k, v, extra_options):
            m = get_functions(q, downsample_factor_depth_1, downsample_factor_depth_2, extra_options["original_shape"])
            return q, m(k), m(v)


        m = model.clone()
        m.set_model_attn1_patch(todo_m)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "ToDoPatchModel": ToDoPatchModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToDoPatchModel" : "ToDo: Token Dowsampling",
}