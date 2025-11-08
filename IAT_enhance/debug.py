# debug_inference_variants.py
import os
import torch
import numpy as np
from PIL import Image
import argparse
from model.IAT_student import IAT_Student_BN
from torchvision import transforms

def to_pil(tensor):
    t = tensor.clamp(0,1).cpu().squeeze(0).permute(1,2,0).numpy()
    return Image.fromarray((t*255).astype(np.uint8))

def save_variant(img_tensor, out_dir, name):
    pil = to_pil(img_tensor)
    pil.save(os.path.join(out_dir, name + '.png'))

def load_ckpt(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        print("⚠️ checkpoint not found:", ckpt_path)
        return False, None
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    print("Checkpoint type:", type(raw))
    if isinstance(raw, dict):
        print("Checkpoint keys:", list(raw.keys()))
        sd = raw.get('state_dict', raw)
    else:
        sd = raw
    # try load safely
    try:
        model.load_state_dict(sd, strict=False)
        print("✅ Loaded checkpoint into model (strict=False).")
        return True, raw
    except Exception as e:
        print("❌ load_state_dict failed:", e)
        # try removing module.
        new_sd = {}
        for k,v in sd.items():
            new_sd[k.replace('module.','')] = v
        try:
            model.load_state_dict(new_sd, strict=False)
            print("✅ Loaded after stripping 'module.' prefix.")
            return True, raw
        except Exception as e2:
            print("❌ still failed:", e2)
            return False, raw

def debug_print_tensor(name, t):
    if isinstance(t, torch.Tensor):
        print(f"{name}: dtype={t.dtype} device={t.device} shape={t.shape} min={t.min().item():.6f} max={t.max().item():.6f} mean={t.mean().item():.6f}")
        if t.numel() <= 20:
            print(f"  values: {t.flatten()[:20].cpu().numpy()}")
    else:
        print(name, type(t))

def load_image_pil(path):
    img = Image.open(path)
    arr = np.asarray(img).astype(np.float32) / 255.0
    # handle grayscale
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return arr

def run_variants(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # model
    model = IAT_Student_BN().to(device)
    loaded, raw_ckpt = load_ckpt(model, args.model_path, device) if args.model_path else (False, None)
    # print a few model param sums to compare random vs loaded
    total_sum = 0.0
    for i, p in enumerate(model.parameters()):
        if i < 5:
            print(f"param[{i}] sum={p.data.sum().item():.6f}")
        total_sum += float(p.data.sum().item())
    print("Model parameter sum preview:", total_sum)

    # load input once as numpy
    arr = load_image_pil(args.input_image)
    print("Input numpy shape, min/max:", arr.shape, arr.min(), arr.max())
    # per-channel mean
    print("Input per-channel mean:", arr[...,0].mean(), arr[...,1].mean(), arr[...,2].mean())

    # variants
    combos = []
    for do_bgr in [False, True]:
        for do_norm in [False, True]:
            combos.append((do_bgr, do_norm))

    # normalization op
    normalize = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))

    for do_bgr, do_norm in combos:
        name = f"BGR_{int(do_bgr)}__NORM_{int(do_norm)}"
        print("\n" + "-"*60)
        print("Running variant:", name)

        arr2 = arr.copy()
        if arr2.shape[2] == 4:
            arr2 = arr2[:, :, :3]

        if do_bgr:
            arr2 = arr2[:, :, ::-1].copy()  # RGB -> BGR

        t = torch.from_numpy(arr2).float().permute(2,0,1).unsqueeze(0).to(device)  # [1,C,H,W]
        debug_print_tensor("input_before_norm", t)

        if do_norm:
            t = normalize(t[0]).unsqueeze(0)
            debug_print_tensor("input_after_norm", t)

        with torch.no_grad():
            out = model(t)
            # model returns (mul, add, img_high) or dict if coded differently
            if isinstance(out, tuple) or isinstance(out, list):
                if len(out) == 3:
                    mul, add, img_high = out
                else:
                    print("Unexpected tuple output len:", len(out))
                    img_high = out[-1]
            elif isinstance(out, dict):
                img_high = out.get('img_high', None)
                mul = out.get('mul', None)
                add = out.get('add', None)
            else:
                print("Unexpected output type:", type(out))
                img_high = out

        if img_high is None:
            print("No img_high obtained, skipping save for this variant.")
            continue

        debug_print_tensor("img_high", img_high)
        # if normalized input, maybe denormalize for visualization
        img_vis = img_high
        if do_norm:
            img_vis = (img_vis * 0.5 + 0.5).clamp(0,1)

        # save
        out_name = os.path.join(args.output_dir, name)
        save_variant(img_vis, args.output_dir, name)
        print("Saved:", out_name + ".png")

        # print gamma/color if available in model outputs (if dict)
        if isinstance(out, dict):
            if 'gamma' in out:
                debug_print_tensor("gamma", out['gamma'])
            if 'color' in out:
                debug_print_tensor("color", out['color'])

    print("\nAll variants done. Compare images in:", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', required=True)
    parser.add_argument('--model_path', required=False, default=None)
    parser.add_argument('--output_dir', default='debug_out')
    args = parser.parse_args()
    run_variants(args)
