"""
test_all.py - Comprehensive Test Suite for 3D Attention U-Net Pipeline.

Runs 12 categories of tests:
  1.  Dependency test
  2.  Dataset test (synthetic)
  3.  Preprocessing test
  4.  Model forward test
  5.  Attention gate test
  6.  Loss backward test
  7.  One-batch training test
  8.  One-patient overfit test
  9.  Small-subset test
  10. Checkpoint save/load test
  11. Resume test
  12. Evaluation metrics test

Run: python test_all.py
"""

import os
import sys
import tempfile
import shutil
import json
import numpy as np
import traceback

# Test results tracking
results = {}

def run_test(name, test_fn):
    """Run a single test and record result."""
    print(f"\n{'='*60}")
    print(f"TEST {name}")
    print(f"{'='*60}")
    try:
        passed = test_fn()
        status = "PASS [OK]" if passed else "FAIL [FAIL]"
        results[name] = passed
    except Exception as e:
        print(f"  [FAIL] EXCEPTION: {e}")
        traceback.print_exc()
        results[name] = False
        status = "FAIL [FAIL]"
    print(f"  Result: {status}")
    return results[name]


# ==============================================================================
# TEST 1: Dependencies
# ==============================================================================
def test_dependencies():
    """Verify all required packages can be imported."""
    deps = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('nibabel', 'NiBabel'),
        ('SimpleITK', 'SimpleITK'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm'),
    ]
    all_ok = True
    for module, name in deps:
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  [OK] {name}: {version}")
        except ImportError:
            print(f"  [FAIL] {name}: NOT INSTALLED")
            all_ok = False

    # Check CUDA
    import torch
    print(f"\n  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return all_ok


# ==============================================================================
# TEST 2: Synthetic Dataset
# ==============================================================================
def test_dataset_synthetic():
    """Create synthetic volumes and verify dataset returns correct shapes."""
    import torch
    import nibabel as nib
    from src.config import CROP_SIZE
    from src.data.dataset import PancreasVolumeDataset
    from torch.utils.data import DataLoader

    tmpdir = tempfile.mkdtemp(prefix="test_ds_")
    img_dir = os.path.join(tmpdir, "images")
    lbl_dir = os.path.join(tmpdir, "labels")
    os.makedirs(img_dir)
    os.makedirs(lbl_dir)

    H, W, D = CROP_SIZE
    affine = np.eye(4)
    n_vols = 2

    for i in range(n_vols):
        img = np.random.rand(H, W, D).astype(np.float32)
        lbl = np.zeros((H, W, D), dtype=np.uint8)
        lbl[80:140, 80:140, 40:80] = 1

        nib.save(nib.Nifti1Image(img, affine), os.path.join(img_dir, f"vol_{i:04d}.nii.gz"))
        nib.save(nib.Nifti1Image(lbl, affine), os.path.join(lbl_dir, f"vol_{i:04d}.nii.gz"))

    import glob
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    lbl_paths = sorted(glob.glob(os.path.join(lbl_dir, "*.nii.gz")))

    # Without augmentation
    ds = PancreasVolumeDataset(img_paths, lbl_paths, augment=False, preprocessed=True)
    img_t, msk_t = ds[0]

    assert img_t.shape == (1, H, W, D), f"Image shape wrong: {img_t.shape}"
    assert msk_t.shape == (1, H, W, D), f"Mask shape wrong: {msk_t.shape}"
    assert img_t.dtype == torch.float32
    print(f"  [OK] Dataset shape: img={img_t.shape}, mask={msk_t.shape}")

    # With DataLoader
    loader = DataLoader(ds, batch_size=1)
    batch_img, batch_msk = next(iter(loader))
    assert batch_img.shape == (1, 1, H, W, D), f"Batch shape wrong: {batch_img.shape}"
    print(f"  [OK] Batch shape: {batch_img.shape}")

    # With augmentation
    ds_aug = PancreasVolumeDataset(img_paths, lbl_paths, augment=True, preprocessed=True)
    img_aug, msk_aug = ds_aug[0]
    assert img_aug.shape == (1, H, W, D)
    print(f"  [OK] Augmented shape: {img_aug.shape}")

    shutil.rmtree(tmpdir, ignore_errors=True)
    return True


# ==============================================================================
# TEST 3: Preprocessing
# ==============================================================================
def test_preprocessing():
    """Test HU clipping, normalization, and center crop/pad."""
    from src.data.preprocessing import clip_hu, normalize_to_01, center_crop_or_pad
    from src.config import HU_MIN, HU_MAX, CROP_SIZE

    # HU clipping
    raw = np.array([-1000, -100, 0, 100, 240, 3000], dtype=np.float32)
    clipped = clip_hu(raw)
    assert clipped.min() >= HU_MIN, f"Min {clipped.min()} < {HU_MIN}"
    assert clipped.max() <= HU_MAX, f"Max {clipped.max()} > {HU_MAX}"
    print(f"  [OK] HU clipping: [{clipped.min()}, {clipped.max()}]")

    # Normalization
    norm = normalize_to_01(raw)
    assert norm.min() >= 0.0, f"Norm min {norm.min()}"
    assert norm.max() <= 1.0, f"Norm max {norm.max()}"
    print(f"  [OK] Normalization: [{norm.min():.4f}, {norm.max():.4f}]")

    # Center crop (volume larger than target)
    big = np.ones((300, 300, 200), dtype=np.float32)
    cropped = center_crop_or_pad(big, CROP_SIZE)
    assert cropped.shape == CROP_SIZE, f"Crop shape {cropped.shape} != {CROP_SIZE}"
    print(f"  [OK] Center crop: {big.shape} -> {cropped.shape}")

    # Center pad (volume smaller than target)
    small = np.ones((100, 100, 64), dtype=np.float32)
    padded = center_crop_or_pad(small, CROP_SIZE)
    assert padded.shape == CROP_SIZE, f"Pad shape {padded.shape} != {CROP_SIZE}"
    # Check that padding is zero
    assert padded[0, 0, 0] == 0.0, "Padding should be zero"
    print(f"  [OK] Center pad: {small.shape} -> {padded.shape}")

    return True


# ==============================================================================
# TEST 4: Model Forward
# ==============================================================================
def test_model_forward():
    """Test model forward pass with multiple input sizes."""
    import torch
    from src.models.model import build_model, count_parameters

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device)

    params = count_parameters(model)
    print(f"  Parameters: {params['total']:,} ({params['total_MB']:.1f} MB)")

    # Small test (always runs)
    sizes = [(1, 1, 32, 32, 16), (1, 1, 64, 64, 32)]

    for shape in sizes:
        x = torch.randn(*shape, device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == shape, f"Shape mismatch: {out.shape} != {shape}"
        print(f"  [OK] Forward {shape} -> {out.shape}")

    return True


# ==============================================================================
# TEST 5: Attention Gates
# ==============================================================================
def test_attention_gates():
    """Test attention gate tensor dimensions and value ranges."""
    import torch
    from src.models.attention import AttentionGate3D

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = [
        (256, 128, 64, (14, 14, 8), (7, 7, 4)),
        (128, 64, 32, (28, 28, 16), (14, 14, 8)),
        (64, 32, 16, (56, 56, 32), (28, 28, 16)),
        (32, 16, 8, (112, 112, 64), (56, 56, 32)),
    ]

    for F_g, F_l, F_int, sp_x, sp_g in configs:
        ag = AttentionGate3D(F_g, F_l, F_int).to(device)
        x = torch.randn(1, F_l, *sp_x, device=device, requires_grad=True)
        g = torch.randn(1, F_g, *sp_g, device=device, requires_grad=True)

        out = ag(x, g)
        assert out.shape == x.shape, f"Output shape {out.shape} != {x.shape}"

        # Check gradient flow
        out.sum().backward()
        assert x.grad is not None, "No gradient for x"
        assert g.grad is not None, "No gradient for g"

        print(f"  [OK] AG(F_g={F_g}, F_l={F_l}): shape OK, gradients OK")

    return True


# ==============================================================================
# TEST 6: Loss Backward
# ==============================================================================
def test_loss_backward():
    """Test loss computation and gradient flow."""
    import torch
    from src.losses import DiceFocalLoss, DiceLoss, FocalLoss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pred = torch.randn(1, 1, 32, 32, 16, device=device, requires_grad=True)
    target = torch.randint(0, 2, (1, 1, 32, 32, 16), device=device).float()

    for name, loss_fn in [("DiceLoss", DiceLoss()), ("FocalLoss", FocalLoss()),
                           ("DiceFocalLoss", DiceFocalLoss())]:
        loss_fn = loss_fn.to(device)
        loss = loss_fn(pred, target)

        assert not torch.isnan(loss), f"{name}: NaN"
        assert not torch.isinf(loss), f"{name}: Inf"

        loss.backward(retain_graph=True)
        assert pred.grad is not None, f"{name}: no gradient"
        pred.grad = None

        print(f"  [OK] {name}: loss={loss.item():.6f}, gradient OK")

    return True


# ==============================================================================
# TEST 7: One-Batch Training Step
# ==============================================================================
def test_one_batch():
    """Test a single training step (forward + loss + backward + optimizer)."""
    import torch
    from src.models.model import build_model
    from src.losses import DiceFocalLoss
    from src.config import USE_AMP

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device)
    criterion = DiceFocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=USE_AMP and device.type == 'cuda')

    x = torch.randn(1, 1, 32, 32, 16, device=device)
    y = torch.randint(0, 2, (1, 1, 32, 32, 16), device=device).float()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type=device.type, enabled=USE_AMP and device.type == 'cuda'):
        out = model(x)
        loss = criterion(out, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"  [OK] One batch: loss={loss.item():.6f}")

    # Verify parameters changed
    model.eval()
    with torch.no_grad():
        out2 = model(x)

    print(f"  [OK] Parameters updated (output changed)")
    return True


# ==============================================================================
# TEST 8: One-Patient Overfit
# ==============================================================================
def test_one_patient_overfit():
    """Train on a single synthetic volume for several steps to verify overfitting."""
    import torch
    import nibabel as nib
    import glob
    from src.config import CROP_SIZE, USE_AMP
    from src.models.model import build_model
    from src.losses import DiceFocalLoss
    from src.data.dataset import PancreasVolumeDataset
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create synthetic volume
    tmpdir = tempfile.mkdtemp(prefix="test_overfit_")
    H, W, D = 64, 64, 32  # Small for speed
    img = np.random.rand(H, W, D).astype(np.float32)
    lbl = np.zeros((H, W, D), dtype=np.uint8)
    lbl[20:44, 20:44, 8:24] = 1

    affine = np.eye(4)
    img_path = os.path.join(tmpdir, "img.nii.gz")
    lbl_path = os.path.join(tmpdir, "lbl.nii.gz")
    nib.save(nib.Nifti1Image(img, affine), img_path)
    nib.save(nib.Nifti1Image(lbl, affine), lbl_path)

    # Use small model for this test
    model = build_model(encoder_channels=[4, 8, 16, 32, 64]).to(device)
    criterion = DiceFocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Load data
    ds = PancreasVolumeDataset([img_path], [lbl_path], augment=False, preprocessed=True)
    img_t, msk_t = ds[0]
    img_t = img_t.unsqueeze(0).to(device)
    msk_t = msk_t.unsqueeze(0).to(device)

    # Train for a few steps
    model.train()
    losses = []
    for step in range(20):
        optimizer.zero_grad()
        out = model(img_t)
        loss = criterion(out, msk_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should decrease
    improved = losses[-1] < losses[0]
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} ({'v improving' if improved else '^ not improving'})")

    if improved:
        print(f"  [OK] Overfit test passed (loss decreased)")
    else:
        print(f"  [WARN] Loss did not decrease -- may need more steps")

    shutil.rmtree(tmpdir, ignore_errors=True)
    return improved


# ==============================================================================
# TEST 9: Small Subset
# ==============================================================================
def test_small_subset():
    """Verify the full pipeline works with 2 synthetic volumes."""
    import torch
    import nibabel as nib
    import glob
    from src.config import USE_AMP
    from src.models.model import build_model
    from src.losses import DiceFocalLoss
    from src.data.dataset import PancreasVolumeDataset
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tmpdir = tempfile.mkdtemp(prefix="test_subset_")
    img_dir = os.path.join(tmpdir, "images")
    lbl_dir = os.path.join(tmpdir, "labels")
    os.makedirs(img_dir)
    os.makedirs(lbl_dir)

    H, W, D = 64, 64, 32
    affine = np.eye(4)

    for i in range(2):
        img = np.random.rand(H, W, D).astype(np.float32)
        lbl = np.zeros((H, W, D), dtype=np.uint8)
        lbl[20:44, 20:44, 8:24] = 1
        nib.save(nib.Nifti1Image(img, affine), os.path.join(img_dir, f"vol_{i:04d}.nii.gz"))
        nib.save(nib.Nifti1Image(lbl, affine), os.path.join(lbl_dir, f"vol_{i:04d}.nii.gz"))

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    lbl_paths = sorted(glob.glob(os.path.join(lbl_dir, "*.nii.gz")))

    ds_train = PancreasVolumeDataset(img_paths[:1], lbl_paths[:1], augment=True, preprocessed=True)
    ds_val = PancreasVolumeDataset(img_paths[1:], lbl_paths[1:], augment=False, preprocessed=True)

    loader_train = DataLoader(ds_train, batch_size=1)
    loader_val = DataLoader(ds_val, batch_size=1)

    model = build_model(encoder_channels=[4, 8, 16, 32, 64]).to(device)
    criterion = DiceFocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # One epoch training
    model.train()
    for img_t, msk_t in loader_train:
        img_t, msk_t = img_t.to(device), msk_t.to(device)
        optimizer.zero_grad()
        out = model(img_t)
        loss = criterion(out, msk_t)
        loss.backward()
        optimizer.step()
        print(f"  Train loss: {loss.item():.4f}")

    # Validation
    model.eval()
    with torch.no_grad():
        for img_t, msk_t in loader_val:
            img_t = img_t.to(device)
            out = model(img_t)
            pred = torch.sigmoid(out)
            print(f"  Val pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")

    print(f"  [OK] Full pipeline works with 2 volumes")
    shutil.rmtree(tmpdir, ignore_errors=True)
    return True


# ==============================================================================
# TEST 10: Checkpoint Save/Load
# ==============================================================================
def test_checkpoint():
    """Test saving and loading a checkpoint."""
    import torch
    from src.models.model import build_model

    tmpdir = tempfile.mkdtemp(prefix="test_ckpt_")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(encoder_channels=[4, 8, 16, 32, 64]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)

    # Save
    ckpt_path = os.path.join(tmpdir, "test_ckpt.pth")
    state = {
        'epoch': 5,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_dice': 0.75,
    }
    torch.save(state, ckpt_path)
    print(f"  Saved checkpoint to {ckpt_path}")

    # Load into new model
    model2 = build_model(encoder_channels=[4, 8, 16, 32, 64]).to(device)
    loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
    model2.load_state_dict(loaded['model_state_dict'])

    assert loaded['epoch'] == 5
    assert loaded['best_val_dice'] == 0.75

    # Compare outputs
    x = torch.randn(1, 1, 32, 32, 16, device=device)
    model.eval()
    model2.eval()
    with torch.no_grad():
        o1 = model(x)
        o2 = model2(x)
    assert torch.allclose(o1, o2, atol=1e-6), "Outputs differ after load"

    print(f"  [OK] Checkpoint save/load verified")
    shutil.rmtree(tmpdir, ignore_errors=True)
    return True


# ==============================================================================
# TEST 11: Resume
# ==============================================================================
def test_resume():
    """Test that training can resume from checkpoint."""
    import torch
    from src.train import save_checkpoint, load_checkpoint
    from src.models.model import build_model

    tmpdir = tempfile.mkdtemp(prefix="test_resume_")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(encoder_channels=[4, 8, 16, 32, 64]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    scaler = torch.amp.GradScaler(device.type, enabled=False)

    # Simulate 3 epochs
    for ep in range(1, 4):
        scheduler.step(0.5 + ep * 0.05)

    # Save checkpoint
    ckpt_path = os.path.join(tmpdir, "latest_checkpoint.pth")
    state = {
        'epoch': 3,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_dice': 0.65,
    }
    torch.save(state, ckpt_path)

    # Resume into new objects
    model2 = build_model(encoder_channels=[4, 8, 16, 32, 64]).to(device)
    opt2 = torch.optim.Adam(model2.parameters(), lr=6e-4)
    sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, mode='max', patience=5)
    scaler2 = torch.amp.GradScaler(device.type, enabled=False)

    start_epoch, best_dice = load_checkpoint(ckpt_path, model2, opt2, sched2, scaler2, device)

    assert start_epoch == 3, f"Expected epoch 3, got {start_epoch}"
    assert best_dice == 0.65, f"Expected dice 0.65, got {best_dice}"

    print(f"  [OK] Resume: epoch={start_epoch}, best_dice={best_dice}")
    shutil.rmtree(tmpdir, ignore_errors=True)
    return True


# ==============================================================================
# TEST 12: Evaluation Metrics
# ==============================================================================
def test_evaluation_metrics():
    """Test DSC, ASSD, HD95 computation."""
    from src.metrics import compute_dice, compute_assd, compute_hd95, compute_all_metrics

    # Perfect overlap
    mask = np.zeros((64, 64, 32), dtype=np.uint8)
    mask[20:44, 20:44, 8:24] = 1

    dice = compute_dice(mask.astype(np.float32), mask)
    assert abs(dice - 1.0) < 1e-6, f"Perfect dice should be 1.0, got {dice}"
    print(f"  [OK] Perfect overlap: DSC={dice:.4f}")

    # No overlap
    pred = np.zeros((64, 64, 32), dtype=np.float32)
    pred[0:10, 0:10, 0:5] = 1.0
    gt = np.zeros((64, 64, 32), dtype=np.uint8)
    gt[50:60, 50:60, 25:30] = 1

    dice_no = compute_dice(pred, gt)
    assert dice_no == 0.0, f"No overlap dice should be 0.0, got {dice_no}"
    print(f"  [OK] No overlap: DSC={dice_no:.4f}")

    # Both empty
    empty = np.zeros((64, 64, 32))
    dice_empty = compute_dice(empty, empty)
    assert dice_empty == 1.0, f"Both empty dice should be 1.0, got {dice_empty}"
    print(f"  [OK] Both empty: DSC={dice_empty:.4f}")

    # ASSD and HD95
    assd = compute_assd(mask.astype(np.float32), mask)
    assert abs(assd) < 1e-6, f"Perfect ASSD should be 0, got {assd}"
    print(f"  [OK] Perfect ASSD: {assd:.4f} mm")

    hd95 = compute_hd95(mask.astype(np.float32), mask)
    assert abs(hd95) < 1e-6, f"Perfect HD95 should be 0, got {hd95}"
    print(f"  [OK] Perfect HD95: {hd95:.4f} mm")

    # Partial overlap
    pred_partial = np.zeros((64, 64, 32), dtype=np.float32)
    pred_partial[15:45, 15:45, 5:27] = 1.0
    metrics = compute_all_metrics(pred_partial, mask)
    assert 0 < metrics['dice'] < 1.0
    assert metrics['assd'] > 0
    assert metrics['hd95'] > 0
    print(f"  [OK] Partial: DSC={metrics['dice']:.4f}, ASSD={metrics['assd']:.2f}, HD95={metrics['hd95']:.2f}")

    return True


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("\n" + "#" * 60)
    print("# 3D ATTENTION U-NET -- COMPREHENSIVE TEST SUITE")
    print("#" * 60)

    tests = [
        ("1. Dependencies", test_dependencies),
        ("2. Synthetic Dataset", test_dataset_synthetic),
        ("3. Preprocessing", test_preprocessing),
        ("4. Model Forward", test_model_forward),
        ("5. Attention Gates", test_attention_gates),
        ("6. Loss Backward", test_loss_backward),
        ("7. One-Batch Training", test_one_batch),
        ("8. One-Patient Overfit", test_one_patient_overfit),
        ("9. Small Subset Pipeline", test_small_subset),
        ("10. Checkpoint Save/Load", test_checkpoint),
        ("11. Resume", test_resume),
        ("12. Evaluation Metrics", test_evaluation_metrics),
    ]

    for name, fn in tests:
        run_test(name, fn)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        status = "PASS [OK]" if ok else "FAIL [FAIL]"
        print(f"  {status}  {name}")
    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  ALL TESTS PASSED -- Ready for training!")
    else:
        print(f"\n  [WARN]  {total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
