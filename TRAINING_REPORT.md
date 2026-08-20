# SimAE v1 — Training Report

**Date:** 
**Runtime:** Colab / Kaggle / Local

---

## Training Config

| Parameter | Value |
|-----------|-------|
| Latent dims | 128 |
| Batch size | 128 |
| Epochs | 30 |
| LR | 1e-3 |
| λ_contrastive | 0.1 |
| Data | Synthetic (n=50000, patterns=5) |

---

## Results

### Training Curves
- Final train reconstruction loss: 
- Final val reconstruction loss: 
- Final contrastive loss: 

### Latent Space
- Did UMAP show clusters? 
- Number of clusters discovered: 
- Do they match known patterns? 

### Reconstruction
- MSE: 
- GC content preserved? 

---

## Verdict

**Success?** Yes / No

**What worked:**
- 

**What didn't:**
- 

---

## Next Steps

- [ ] Try on real genomic data
- [ ] Adjust contrastive weight
- [ ] Try smaller/larger latent
- [ ] Add more patterns to synthetic data
