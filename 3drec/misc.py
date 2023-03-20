import numpy as np
import torch


def torch_samps_to_imgs(imgs, uncenter=True):
    if uncenter:
        imgs = (imgs + 1) / 2  # [-1, 1] -> [0, 1]
    imgs = (imgs * 255).clamp(0, 255)
    imgs = imgs.to(torch.uint8)
    imgs = imgs.permute(0, 2, 3, 1)
    imgs = imgs.cpu().numpy()
    return imgs


def imgs_to_torch(imgs):
    assert imgs.dtype == np.uint8
    assert len(imgs.shape) == 4 and imgs.shape[-1] == 3, "expect (N, H, W, C)"
    _, H, W, _ = imgs.shape

    imgs = imgs.transpose(0, 3, 1, 2)
    imgs = (imgs / 255).astype(np.float32)
    imgs = (imgs * 2) - 1
    imgs = torch.as_tensor(imgs)
    H, W = [_l - (_l % 32) for _l in (H, W)]
    imgs = torch.nn.functional.interpolate(imgs, (H, W), mode="bilinear")
    return imgs


def test_encode_decode():
    import imageio
    from run_img_sampling import ScoreAdapter, SD
    from vis import _draw

    fname = "~/clean.png"
    raw = imageio.imread(fname)
    raw = imgs_to_torch(raw[np.newaxis, ...])

    model: ScoreAdapter = SD().run()
    raw = raw.to(model.device)
    zs = model.encode(raw)
    img = model.decode(zs)
    img = torch_samps_to_imgs(img)
    _draw(
        [imageio.imread(fname), img.squeeze(0)],
    )


def test():
    test_encode_decode()


if __name__ == "__main__":
    test()
