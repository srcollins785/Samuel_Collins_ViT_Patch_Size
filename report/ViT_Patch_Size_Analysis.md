# Patch Size in a Small Vision Transformer: Analysis

**Samuel Collins** - Ph.D. Student, Department of Cyber-Physical Systems, Clark Atlanta University

Computer Vision - Lab Assignment 6: Vision Transformers (ViT)

Instructor: Dr. Kishor Datta Gupta

Date: 2026-10-16

---

## Analysis

**Performance.** The 4x4 model finished at 38.1% validation accuracy and the 8x8 model at 34.8%, with validation loss 1.673 against 1.758. One pair is thin evidence, since one accuracy carries about 1.1 points of sampling error, but across 5 seeds 4x4 led every time, by 2.5 points on average. The advantage concentrated in particular classes: dog accuracy was 65% against 44%. Most examples behave alike: both models were right on 508 images and wrong on 1,050 of 2,000, and they split on the remainder (254 only 4x4, 188 only 8x8). Smaller patches did not buy accuracy with time: runtimes were 4.1 s and 4.7 s, within the run-to-run noise of about a second, although 4x4 scores 14.6 times as many attention entries. The cost of the larger map is real but invisible at this scale.

**Global attention and position.** A convolution sees a small neighborhood, and distant pixels meet only after many layers; lecture L2 counts O(n/k) layers for kernel size k, against O(1) for self-attention. Here every one of the 65 or 17 tokens attends to all others in the first block. A convolution also shares weights across positions, which builds in translation equivariance; a ViT has no such prior. Attention alone ignores order, so the position embedding is what lets the model use where a patch is; a test confirmed that without it, shuffling patches changes nothing.

**Poor absolute performance.** Both models are far from converged: training accuracy scored in evaluation mode was 40.3% for 4x4, barely above its validation accuracy, and the curves were still falling at epoch five. That is underfitting, not overfitting. A ViT lacks a CNN's built-in locality, so it must learn such structure from data, and the lecture attributes ViT's success to pretraining on hundreds of millions of images; this model saw ten thousand, briefly. The graded experiment ran no CNN, one dataset and one tiny budget, so it says nothing about CNNs or transformers being better in general; the supplementary runs below add a CNN but share those limits.

**Swin for dense tasks.** Three separate ideas help. Windowed attention cuts cost: on a 3,136-token map (56 x 56, a 224-pixel image with 4x4 patches), global attention needs 9,834,496 entries per head, while 7 x 7 windows need 153,664, 64 times fewer; that is arithmetic, not a measurement. Shifted windows restore communication, because tokens in different windows interact in alternate layers. Patch merging builds multiscale features, giving the detector or segmenter the coarse-to-fine maps a flat coarse-patch ViT lacks; lecture L2 has the four stages shrink a 224-pixel input to 56, 28, 14 and 7 tokens per side, the resolutions of ResNet-style backbones. Small objects need fine patches, and windows make fine patches affordable.

**Additional controlled experiment.** The most informative addition is training to convergence with a parameter-matched CNN as a reference, so I ran it (section 4.6 of the full report): 5 seeds, the same split and settings, validation only. The 4x4 lead grew from 2.5 points at five epochs to 10.1 at each run's best epoch, in 5 of 5 seeds. Under this setup that points to a lower ceiling for 8x8 rather than a slower start: it plateaued near 47% against 57% for 4x4. The CNN reached 63%, ahead of both ViTs in every seed, though its margin over 4x4 narrowed from 13.6 points at epoch 5 to 5.4 at epoch 30. The stopping epoch was chosen on validation images, so these accuracies are slightly optimistic; the next steps are a final score on the untouched test split, augmentation and pretraining.

---

All figures, tables, the model description and the sources are in `ViT_Patch_Size_Report.pdf`, and the executed code is in `ViT_Patch_Size.ipynb`. Repository: https://github.com/srcollins785/Samuel_Collins_ViT_Patch_Size
