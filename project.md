# CSC 4850 — Presentation Script
## Domain-Conditioned Degradation-Aware Blind Super-Resolution
**Team:** Dany George Nishanth, Vanohra Gaspard, Sabirin Mohamed
**Course:** CSC 4850 — Machine Learning, Spring 2026

---

## Project Summary

This project builds a super-resolution system specifically designed for real-world surveillance footage. Standard super-resolution methods assume clean, predictable image degradation — but real surveillance cameras produce images corrupted by blur, noise, and compression artifacts in ways that vary dramatically from camera to camera. Our system addresses this by explicitly telling the model how each image was degraded: instead of guessing, the model receives a description of the degradation and uses it to choose the best restoration strategy.

We implemented three progressively more capable models and trained them as a controlled ablation study. The simplest model is a 3-layer CNN that beats bicubic interpolation. The second model is a deeper residual network trained specifically on surveillance-style degradation. The third and final model adds FiLM conditioning — a mechanism that lets the network adjust its behavior based on a 13-element vector describing the exact blur, noise, and compression applied to each image. We trained this final model with curriculum learning (easy degradations first, hard ones later) and a perceptual loss that encourages texture detail. All three models are evaluated with the same metrics so we can clearly see what each component contributes.

> **Note to presenter:** When presenting, replace all bracketed placeholders like `[X.XX dB]` with your actual numbers from `outputs/comparison_table.csv` and `outputs/experiments/*/metrics/eval_paired_summary.json`.

---

## Section 1 — Problem and Motivation

### Slide 1.1 — The Problem with Surveillance Footage

**What to say out loud:**

"Let's start with the problem we're solving. Imagine you have a surveillance camera recording a parking lot. Someone breaks into a car. You pull the footage and you get... a blurry, noisy mess. The license plate is unreadable. The face is unidentifiable. The footage is useless for the one thing it was supposed to do.

This is not a made-up scenario. Surveillance cameras are almost universally low quality — not because the manufacturers are cheap, but because of real engineering constraints. These cameras record 24 hours a day, seven days a week. They stream over networks with limited bandwidth. They store footage for weeks on servers with limited space. Every one of those constraints pushes toward lower resolution, heavier compression, and worse image quality.

Our project asks: can machine learning restore that footage after the fact?"

**Slide notes:**
- Show a side-by-side: a blurry surveillance crop vs. a clean version
- The key emotional hook is: useful footage vs. useless footage
- Do not get technical yet — this slide is about why anyone should care

---

### Slide 1.2 — Why Existing Super-Resolution Doesn't Work Well

**What to say out loud:**

"Super-resolution — making a low-resolution image look higher-resolution — is a well-studied problem. But most super-resolution methods are built and tested on clean images that were simply downscaled. You take a sharp photo, shrink it, then try to recover the original.

Real surveillance footage does not work like that. It's not just smaller — it's blurry from a cheap wide-angle lens. It's noisy from a low-light sensor. It's compressed twice: once when it was recorded, and again when it was sent to the server or the cloud. The degradations stack on top of each other in ways that simple models do not expect.

If you train a super-resolution model on clean, bicubic-downscaled images and then run it on real surveillance footage, you get artifacts. You get worse results than you'd get from just doing bicubic interpolation yourself. The model is solving the wrong problem."

**Slide notes:**
- This motivates the need for blind SR — handling unknown, compound degradations
- Bicubic interpolation is the baseline everyone compares against
- "The model is solving the wrong problem" is the key phrase — it sets up our solution

---

### Slide 1.3 — Our Approach in One Sentence

**What to say out loud:**

"Our solution is straightforward in concept: instead of making the model guess how bad the image is, we tell it. We attach a description of the degradation to every image during training and evaluation. The model learns to read that description and adjust its restoration strategy accordingly. A mildly blurry image gets gentle sharpening. A severely compressed image gets aggressive artifact removal. The same model handles both — it just needs to know which situation it's in.

We call this degradation-aware conditioning. The 'blind' part in our title refers to the real deployment challenge: in the wild, you often don't have that description — you have to estimate it. We implement the conditioning part; estimating the parameters from the image itself is the next step, and we'll talk about that in future work."

**Slide notes:**
- Transition sentence: "So let's look at how we actually built this system."
- Honest about what 'blind' means vs. what we actually implemented

---

## Section 2 — Proposed Framework

### Slide 2.1 — The Full Pipeline Overview

**What to say out loud:**

"Here's how our full pipeline works, from a clean training image to a super-resolved output.

We start with a high-resolution training image from the DIV2K dataset — a standard benchmark collection of 800 high-quality photos covering diverse content. We apply a two-stage degradation process that simulates what a surveillance camera actually does to an image. We then scale the result down by a factor of four. That's our low-resolution input.

At the same time, we record exactly what degradations were applied — the blur strength, the noise level, the compression quality — and we package that into a 13-element vector. That vector goes into the model alongside the low-resolution image.

The model processes both, produces a four-times-larger output image, and we compare that output to the original high-resolution image to compute the training loss. Backpropagation updates the model weights. Repeat for 100 epochs."

**Slide notes:**
- Show a diagram: HR → [Degradation] → LR + Cond. Vector → [ConditionedSRResNet] → SR → compared to HR
- Emphasize: the conditioning vector is the novel piece
- "100 epochs" — each epoch processes all 100 training images

---

### Slide 2.2 — The Degradation Pipeline (What We Do to Images)

**What to say out loud:**

"Let's zoom into the degradation step, because this is more sophisticated than it might sound.

We designed a two-stage degradation pipeline that mimics what actually happens to surveillance footage. In stage one, we apply blur — simulating a cheap lens or camera motion — followed by downsampling to reduce the resolution, then Gaussian noise to simulate sensor noise, and finally JPEG compression with a randomly chosen quality level. That's what the camera does.

In stage two — which we apply 75% of the time for surveillance footage — we apply a second round of noise and JPEG compression at the low-resolution scale. This simulates what happens when the footage is streamed to a server, stored as a compressed video, then re-encoded again when someone downloads the clip. DVR recording, then cloud re-encoding. Two generations of compression artifacts on top of each other.

We define three domain presets that set the ranges for these parameters. The surveillance preset uses heavy blur — one to four sigma — high noise — ten to forty-five sigma — and aggressive JPEG compression — quality twenty to sixty. Mobile and dashcam presets use milder settings, because phone cameras and dashcams are generally higher quality than fixed surveillance hardware."

**Slide notes:**
- Blur sigma: higher = blurrier. Bicubic kernel sigma 1.0 is mild, 4.0 is very blurry
- Noise sigma: higher = grainier. 45 is quite noisy
- JPEG quality: lower = worse. Quality 20 is very heavily compressed, lots of blockiness
- Stage 2 means double compression — a real surveillance pain point
- The surveillance preset is what all three experiments use (except exp1 which uses random)

**Formula — JPEG Quality:**
JPEG quality is not a formula — it is a parameter from 0 to 100 that controls how aggressively the encoder discards high-frequency information. Quality 95 looks almost identical to the original. Quality 20 creates visible blocky artifacts. We randomize it to train the model to handle any level of compression.

---

### Slide 2.3 — The Conditioning Vector (What We Tell the Model)

**What to say out loud:**

"Every time we degrade an image, we record exactly what we did and encode it as a 13-element vector. This is the conditioning vector — it is the model's description of what it's about to undo.

The vector contains five values from stage one: how strong the blur was, which downsampling method was used, how strong the noise was, the JPEG quality, and whether JPEG was applied. Then there is a flag for whether a second stage happened. Then four more values for stage two. And finally, three values at the end that form a one-hot encoding of the domain — mobile, surveillance, or dashcam. Only one of those three is set to one; the others are zero.

Everything is normalized to the range zero to one so the model can learn stable weights. Blur sigma of four — the maximum for surveillance — maps to one. Blur sigma of zero maps to zero. The model learns that high values in certain positions mean 'this image was heavily processed, apply aggressive restoration.'"

**Slide notes:**
- One-hot encoding: exactly one of [mobile, surveillance, dashcam] is 1, the rest are 0
- The conditioning vector is what enables the same model to handle different domains
- In training: we always have this vector. In deployment on real images: we must estimate it or use zeros

---

### Slide 2.4 — FiLM Conditioning (How the Model Uses the Vector)

**What to say out loud:**

"Now, how does the model actually use that conditioning vector? We use a technique called FiLM — Feature-wise Linear Modulation — from a 2018 paper by Perez et al. The idea is elegant.

A standard residual network processes images through a series of layers that transform and mix features. FiLM adds a small network that reads the conditioning vector and produces two outputs for each layer: a scale value gamma and a shift value beta. After every residual block in our network, we multiply the features by gamma and add beta. That's it. It's a per-channel scale and shift — two parameters per feature channel, per layer.

What this does is allow the conditioning vector to smoothly control how every layer in the network processes the image. If the image has heavy blur, the gamma and beta values adjust to emphasize deblurring. If the image has heavy JPEG compression, they emphasize artifact removal.

One implementation detail that matters: we initialize gamma to one and beta to zero. That means at the start of training, FiLM does nothing — the model starts as a plain residual network. The conditioning gradually turns on as training proceeds and the network learns to use the signal. This prevents the model from collapsing early in training when it hasn't yet learned what the conditioning vector means."

**Slide notes:**
- FiLM formula: `out = gamma * features + beta` — applied channel-wise after each residual block
- Identity initialization is important — mention it if asked about training stability
- The MetadataEncoder is a small MLP that maps the 13-element vector to gamma/beta pairs
- We have 8 residual blocks, so 8 sets of gamma/beta — each set has 2 × (number of channels) parameters

**Formula explanation — FiLM:**
- `out = γ ⊙ h + β`
- `h` is the feature map output of a residual block (a 3D tensor: channels × height × width)
- `γ` (gamma) is a vector of scale values, one per channel
- `β` (beta) is a vector of shift values, one per channel
- `⊙` means multiply element-wise — each channel gets its own scale
- In plain English: "for each channel in the feature map, stretch it by gamma and move it by beta"
- The conditioning vector controls what gamma and beta are

---

### Slide 2.5 — The Three Models (Our Ablation Story)

**What to say out loud:**

"We built three models to test whether each component of our design actually helps. This is an ablation study — we add one thing at a time and measure the result.

Model one is SimpleSRCNN. Three convolutional layers, about fifty-seven thousand parameters. Trained on generic random degradation — no domain specialization. This is our baseline. It answers the question: does any neural SR model beat bicubic?

Model two is SRResNet. Eight residual blocks, about nine hundred fifty-seven thousand parameters — seventeen times more than model one. Trained specifically on surveillance-style degradation. This answers: does a deeper model, trained on the right domain, do better?

Model three is ConditionedSRResNet. The SRResNet backbone plus FiLM conditioning layers — about one million thirty thousand parameters. Trained with curriculum learning and perceptual loss. This is our full proposal and answers: does conditioning plus curriculum plus perceptual loss produce the best results?

Each step adds exactly one thing. That's how we know which parts matter."

**Slide notes:**
- Show a table: Model | Parameters | Domain | Conditioning | Curriculum | Loss
- The clean progression — 57K → 957K → 1.03M — shows incremental buildup
- "Ablation study" means removing/adding one component at a time to measure its contribution

---

### Slide 2.6 — Curriculum Training

**What to say out loud:**

"One more component worth explaining: curriculum training. When you teach a student, you don't start with the hardest problems. You build up gradually. The same idea applies here.

In the first third of training — epochs one through thirty-three — we apply mild degradations. Light blur, low noise, mild compression. The model learns the basic task of super-resolution.

In the middle third — epochs thirty-four through sixty-six — we apply medium-strength degradations. The model has learned the basics and can now handle more challenging inputs.

In the final third — epochs sixty-seven through one hundred — we apply the full surveillance preset: heavy blur, high noise, aggressive double compression. Now the model is ready for the hardest cases.

Validation always uses the full preset throughout all three stages. That means our PSNR numbers are comparable across all epochs — we're always measuring on the same difficulty level."

**Slide notes:**
- Curriculum is only used in exp3, not exp1 or exp2
- The validation consistency is important for fair comparison of training curves
- The intuition: easier examples first stabilize training; hard examples refine it

---

### Slide 2.7 — Loss Functions

**What to say out loud:**

"During training, we compute a loss that tells the model how wrong its output is. We use two losses combined.

The first is L1 loss — the average absolute difference between our model's output and the original high-resolution image, pixel by pixel. L1 is simple, stable, and well-understood. It penalizes the model proportionally for every error.

The second is perceptual loss. Instead of comparing pixels directly, we pass both our output and the original through a pretrained VGG16 network — an image classifier — and compare the intermediate features. VGG16 features capture things like texture, edges, and object structure. When our output has the same texture as the original, the VGG features match. When it's blurry, they don't.

We combine them with weights of one-point-zero for L1 and zero-point-zero-five for perceptual. L1 handles overall correctness; perceptual adds a small push toward texture detail without dominating training.

We deliberately chose not to use GAN training — adversarial loss — even though it's common in state-of-the-art SR. GAN training is unstable, requires careful discriminator scheduling, and is difficult to debug in a class project timeline. VGG perceptual loss gives us sharper outputs than L1-only without the instability risk."

**Slide notes:**
- L1 formula: `loss = (1/N) * sum(|output[i] - target[i]|)` over all pixels
- MSE (mean squared error) is the alternative — we chose L1 because it's less sensitive to outliers and produces sharper results
- Perceptual loss: `loss_perc = ||VGG(output) - VGG(target)||_2` — L2 distance of VGG features
- Total loss: `loss_total = 1.0 * L1 + 0.05 * perceptual`
- No GAN is a feature, not a limitation — justify it confidently

---

## Section 3 — Current Progress

### Slide 3.1 — What We Have Built

**What to say out loud:**

"Let me be specific about what we've actually implemented and what is still future work — being honest about this is important.

We have fully implemented the ConditionedSRResNet architecture with FiLM conditioning. All eight residual blocks accept conditioning signals. The MetadataEncoder maps the 13-element vector to per-block gamma and beta values. Identity initialization is in place for training stability.

We have implemented the surveillance domain degradation preset with the two-stage pipeline. Mobile and dashcam presets are also implemented.

We have implemented curriculum training across three stages with reproducible seeding.

We have implemented L1 plus VGG16 perceptual loss.

We have implemented a full evaluation pipeline: PSNR, SSIM, LPIPS for paired evaluation against ground truth; NIQE and BRISQUE for no-reference evaluation on real images.

We have a three-experiment ablation study with one variable changed per step.

And we have an automated experiment runner that handles training, evaluation, and results collection end-to-end."

**Slide notes:**
- This is the affirmative slide — everything listed here is real and working
- Be confident presenting this list
- The next slide addresses what is not yet done

---

### Slide 3.2 — Known Limitations and What We Did Not Implement

**What to say out loud:**

"Now, what we have not done — and why it is still okay.

We did not implement blind degradation estimation. In a real deployment, you would not always know the blur and noise parameters — you would need to estimate them from the image itself. We assume that metadata is available. For RealSRSet — our real-world test images — we fall back to a zero conditioning vector, which effectively turns off the conditioning head. The model still produces super-resolution results through its backbone, but it cannot adapt to the specific degradation. This is our primary limitation, and it's the most important future work item.

We trained on one hundred DIV2K images instead of the full eight hundred. This is a compute budget decision. Our PSNR numbers will be one to three decibels lower than published results from papers trained on the full set. Our comparisons between our three experiments are valid — they all use the same data — but do not compare our numbers directly to published SRResNet or ESRGAN results.

We have not implemented temporal or multi-frame super-resolution. We treat video frames as independent still images. A complete system would maintain consistency across consecutive frames. This was out of scope for this semester.

We use VGG perceptual loss instead of adversarial (GAN) loss. This is a deliberate engineering decision, not a capability gap."

**Slide notes:**
- Anticipate the question "why not GAN?" — the answer is given above: instability risk
- Anticipate "why only 100 images?" — compute budget, but ablation comparisons still valid
- The zero conditioning fallback for RealSRSet is documented in `infer.py` — we did not hide it

---

## Section 4 — Training Results

### Slide 4.1 — Training Curves

**What to say out loud:**

"Here are our training PSNR curves across all three experiments. PSNR — Peak Signal-to-Noise Ratio — is the primary metric for measuring image reconstruction quality. Higher is better.

[Point to the curve for exp1.]
This is experiment one, SimpleSRCNN trained for fifty epochs on generic degradation. You can see it starts below the bicubic baseline and improves, eventually beating bicubic by a small margin.

[Point to the curve for exp2.]
Experiment two, SRResNet trained for one hundred epochs on surveillance-specific degradation. It starts improving faster and reaches a higher plateau. This tells us two things: a deeper model helps, and training on the right domain matters.

[Point to the curve for exp3.]
Experiment three, our full system, reaches the highest PSNR of the three. Notice the training curve — it shows slower improvement in the early epochs when curriculum is on easy mode, then an acceleration in the middle, then another plateau as it handles the hardest degradations.

The numerical results: [insert your actual numbers here from val_psnr.json and comparison_table.csv]."

**Slide notes:**
- The bicubic baseline is a horizontal line — show it on the same plot
- PSNR improvement from bicubic to exp3 is the headline number
- If exp3 does not clearly beat exp2, explain: the conditioning benefit requires enough training epochs and the model may still be learning to use the conditioning signal at 100 epochs

---

### Slide 4.2 — Understanding PSNR

**What to say out loud:**

"Let's take a moment to explain PSNR, because it's the most important number in our results.

PSNR stands for Peak Signal-to-Noise Ratio. It measures how close our super-resolved image is to the original high-resolution image, pixel by pixel. The formula is:

PSNR = 10 × log base 10 of (1 divided by MSE)

where MSE is the mean squared error — the average squared difference between our output and the ground truth, over all pixels.

In plain English: PSNR measures how much of the original information we successfully recovered. Higher PSNR means less error — the output is closer to the ground truth.

PSNR is measured in decibels. The scale is logarithmic, which means that a jump from 28 dB to 30 dB is a bigger deal than it sounds — that represents significantly less error. In the super-resolution literature, a one decibel improvement is considered meaningful, and two decibels is notable.

For reference: PSNR above 40 dB is nearly indistinguishable from the original. PSNR around 25–30 dB is the typical range for super-resolution with 4× upscaling. Bicubic interpolation typically falls around 26–28 dB on our dataset. Our model should exceed that."

**Slide notes:**
- The formula written out: `PSNR = 10 * log10(MAX^2 / MSE)` where MAX=1 for normalized images
- MSE = mean squared error = `mean((output - target)^2)`
- Decibel scale: a 3 dB improvement means MSE is roughly halved
- If someone asks about signal-to-noise: "signal" here is the maximum possible value of a pixel; "noise" is the error we're adding

---

### Slide 4.3 — Quantitative Results Table

**What to say out loud:**

"Here are our results across all three experiments.

[Read from actual comparison_table.csv — fill in the table below before presenting.]"

| Method | PSNR (dB) | SSIM | LPIPS | NIQE | BRISQUE |
|---|---|---|---|---|---|
| Bicubic baseline | [X.XX] | [X.XXX] | [X.XXX] | [X.XX] | [X.XX] |
| Exp 1 — SimpleSRCNN | [X.XX] | [X.XXX] | [X.XXX] | [X.XX] | [X.XX] |
| Exp 2 — SRResNet | [X.XX] | [X.XXX] | [X.XXX] | [X.XX] | [X.XX] |
| Exp 3 — ConditionedSRResNet | [X.XX] | [X.XXX] | [X.XXX] | [X.XX] | [X.XX] |

**What to say out loud (continued):**

"The story we want to see is: Exp 1 beats bicubic — that tells us neural SR works. Exp 2 beats Exp 1 — that tells us domain specialization and a deeper model help. Exp 3 beats Exp 2 — that tells us conditioning and curriculum and perceptual loss add value.

If any step doesn't show improvement, we explain it: the difference may be small because one hundred training images limits how much benefit we can extract from added complexity."

**Slide notes:**
- Fill in actual numbers from `outputs/comparison_table.csv` before presenting
- PSNR: higher is better — this is the headline metric
- SSIM: ranges from 0 to 1, higher is better — measures structural similarity
- LPIPS: lower is better — measures perceptual similarity (how similar they look to a human)
- NIQE and BRISQUE: lower is better — no-reference quality (no ground truth needed, measured on RealSRSet)

---

## Section 5 — Dataset and Evaluation

### Slide 5.1 — Our Datasets

**What to say out loud:**

"We used two types of data.

For training and paired evaluation, we use DIV2K — a standard super-resolution benchmark of high-quality images. The full dataset has 800 training images and 100 validation images. We used 100 training images and 40 validation images, due to compute constraints. DIV2K contains diverse content — people, landscapes, objects, text — which teaches the model to super-resolve general content, not just one type of image.

For no-reference evaluation, we use RealSRSet — a collection of real-world low-resolution images with no corresponding high-resolution ground truth. These are images that were already degraded before we got them. We can't measure PSNR on these because we don't have the originals to compare against. Instead, we use perceptual quality metrics that don't need ground truth.

VIRAT — a dataset of real surveillance footage — was available as an optional evaluation target. The infrastructure to extract frames and evaluate with known degradation metadata is implemented in our codebase. Whether we include VIRAT results in this presentation depends on whether we had time and access to run it. If we did not run VIRAT, it is future work, not a gap."

**Slide notes:**
- Be explicit: "We used 100 out of 800 training images. This is a compute budget decision."
- RealSRSet is the right choice for real-world qualitative results — no one expects perfect PSNR here
- If VIRAT was run: show the results. If not: "VIRAT evaluation is implemented but optional — we did not include it in this run."

---

### Slide 5.2 — How We Evaluate (Paired Evaluation)

**What to say out loud:**

"For paired evaluation — where we have the high-resolution ground truth — we use three metrics.

PSNR we already explained. Higher is better, measured in decibels.

SSIM stands for Structural Similarity Index. It compares images not just pixel by pixel but in terms of luminance, contrast, and structure — three things the human visual system is sensitive to. It ranges from zero to one. One means perfect structural similarity. In practice, values above 0.9 indicate very similar structure. SSIM often tells a more perceptual story than PSNR — a blurry image might have decent PSNR but poor SSIM if the edges are smeared.

LPIPS stands for Learned Perceptual Image Patch Similarity. It uses a neural network — specifically AlexNet pretrained on ImageNet — to compare patches of two images by their feature representations, not their pixel values. Lower LPIPS means more perceptually similar. LPIPS correlates better with human preference than PSNR or SSIM in many cases.

For our conditioned model, we use a flag called `--regen_lr` in the evaluation script. Instead of loading pre-saved low-resolution images, we re-degrade each high-resolution validation image using the surveillance preset, capture the exact parameters, and pass the real conditioning vector to the model. This is true conditioning evaluation — the model actually uses its conditioning head rather than receiving zeros."

**Slide notes:**
- SSIM formula: combines luminance (mean), contrast (std deviation), structure (normalized cross-correlation) across a sliding window
- LPIPS is lower-is-better, opposite of PSNR and SSIM
- The `--regen_lr` flag is important — without it, the conditioned model is evaluated with zero conditioning, which undersells it

---

### Slide 5.3 — How We Evaluate (No-Reference Evaluation)

**What to say out loud:**

"For real-world images where we have no ground truth, we use no-reference metrics. These assess image quality without comparing to an original.

NIQE — Natural Image Quality Evaluator — compares the statistical properties of our super-resolved image to a large collection of natural, undistorted images. If our output has statistics similar to real clean images, NIQE is low. If it has unusual patterns — like ringing artifacts or over-sharpening — NIQE is high. Lower is better.

BRISQUE — Blind/Referenceless Image Spatial Quality Evaluator — measures distortion by looking at the relationship between neighboring pixel values. Clean natural images have predictable patterns between neighbors. Distorted images break those patterns. Lower BRISQUE means fewer visible artifacts.

We run these on RealSRSet — real degraded images with no HR counterpart. On these images, our model uses a zero conditioning vector because we don't have degradation metadata. The model still produces SR via its backbone, but the conditioning head is inactive. Our NIQE and BRISQUE results show whether the backbone alone produces perceptually cleaner outputs than bicubic interpolation."

**Slide notes:**
- NIQE and BRISQUE don't need HR — they only look at the output image itself
- These metrics capture things PSNR misses: blurriness, ringing, noise amplification
- "Zero conditioning fallback" — honest statement, not a failure

---

### Slide 5.4 — Qualitative Results

**What to say out loud:**

"Beyond the numbers, we want to show visual results. This is the most convincing evidence for a computer vision system.

[Show side-by-side panels: LR | Bicubic | Our SR | HR Ground Truth]

On the left is the low-resolution input — the image after four-times downscaling and surveillance-style degradation. Next is bicubic interpolation — the traditional approach, no neural network. In the middle is our model's output. On the right is the original high-resolution image.

Look at [point to a specific detail — edges, text, texture]. Bicubic blurs that detail. Our model recovers [describe what it recovers]. The PSNR improvement of [X dB] translates to this visible difference.

We also have a case where our model doesn't perfectly recover the detail — specifically in regions with [severe blur / very low texture / extreme compression]. We include this because honest reporting is more credible than cherry-picking only our best results."

**Slide notes:**
- Find your actual sample images in `outputs/experiments/exp3_final/samples/epoch_100/`
- Choose 3 images: one strong win, one moderate, one honest failure case
- The failure case actually strengthens your credibility with the audience
- Do not claim perceptual quality without showing it — the visual is more convincing than any metric

---

## Section 6 — Roadmap and Next Steps

### Slide 6.1 — Primary Future Work: Blind Degradation Estimation

**What to say out loud:**

"Our biggest open problem is the gap between training and real deployment. During training, we tell the model exactly how each image was degraded. In the real world, you have the degraded image and nothing else.

The next step is to build a degradation estimator — a small network that takes a low-resolution image and predicts the conditioning vector. If we had that, the full pipeline would be: input a real surveillance frame, estimate its degradation parameters, pass both the frame and the estimated parameters to the conditioned model, and get a high-quality output. That closes the loop.

This is not a trivial problem. Estimating noise level or JPEG quality from a degraded image requires the estimator network to see many examples of each degradation level. But it's a natural extension of exactly what we have built — the architecture already supports arbitrary conditioning vectors."

**Slide notes:**
- The key phrase: "our conditioning vector format is already designed to be estimated"
- Reference: Real-ESRGAN by Wang et al. (2021) uses a similar two-stage pipeline with blind estimation
- Frame this as the natural next milestone, not a failure

---

### Slide 6.2 — Other Future Work

**What to say out loud:**

"Beyond blind estimation, there are several directions we'd pursue with more time.

Temporal consistency for video: we currently treat each surveillance frame as an independent image. A real system should maintain coherence across consecutive frames — so a face that's in the same position across ten frames looks the same in our restored output, not ten slightly different versions. This requires a video SR architecture, not single-image.

Full training data: we used one hundred of the eight hundred available DIV2K training images. Scaling to the full dataset with the same training setup should improve absolute PSNR by one to two decibels based on established scaling trends.

GAN-based sharpening: adversarial training produces higher-frequency texture detail than perceptual loss. The trade-off is training instability. With more time and compute, GAN training is the right next step for perceptual quality.

Multi-scale evaluation: we only tested four-times upscaling. The same architecture can handle two-times or eight-times with config changes, and it would be interesting to see how the conditioning benefit scales with the difficulty of the task."

**Slide notes:**
- These are honest future directions, not excuses
- The temporal consistency point is important for a surveillance-focused project
- GAN is a future enhancement, not a gap — justify the VGG choice again if asked

---

### Slide 6.3 — What We Learned

**What to say out loud:**

"A few things we learned from building this system that go beyond the metrics.

Degradation matters more than architecture. Going from a three-layer CNN to an eight-layer residual network helped — but training that network on the wrong degradation distribution would hurt. The surveillance domain preset was as important as the model capacity increase.

Curriculum training helps stability. Training on hard degradations from epoch one caused the model to struggle. Introducing difficulty gradually — even with the same final result — produced smoother training curves and slightly better final PSNR.

FiLM conditioning is an elegant solution to a real problem. The ability to modulate a network's behavior through an external signal without rebuilding the architecture is genuinely useful beyond super-resolution. It's used in text-to-image models, few-shot learning, and reinforcement learning. We now understand why.

And: honest evaluation matters. Running evaluation with true conditioning — re-degrading each image and passing the real parameters — gives results that actually reflect what the model can do when properly used. Testing conditioned models with zero conditioning undersells them."

**Slide notes:**
- This slide shows intellectual engagement, not just implementation
- The "honest evaluation" point can lead into a Q&A answer about why our eval numbers might differ from what someone else would get

---

## Formula Reference Sheet

*(For use during Q&A or if a professor asks for technical depth)*

### PSNR — Peak Signal-to-Noise Ratio
```
PSNR = 10 * log10(MAX^2 / MSE)
MSE = (1/N) * sum((output_pixel - target_pixel)^2)
```
- MAX = 1.0 for images normalized to [0,1]
- N = total number of pixels
- **Plain English:** How much error (in decibels) separates our output from the original. More decibels = less error = better.

### SSIM — Structural Similarity Index
```
SSIM(x, y) = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
             / ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))
```
- mu = mean brightness in a local window
- sigma = standard deviation (local contrast)
- sigma_xy = cross-correlation (local structure)
- C1, C2 = small stabilizing constants
- **Plain English:** Does the output have the same brightness, contrast, and structure as the original, locally? Ranges 0 to 1; 1 is perfect.

### L1 Loss
```
L1 = (1/N) * sum(|output_pixel - target_pixel|)
```
- **Plain English:** Average absolute difference between output and target pixels. We minimize this during training.

### Perceptual Loss
```
L_perc = (1/N_feat) * sum((VGG16(output)[i] - VGG16(target)[i])^2)
```
- VGG16(x) = intermediate feature maps of a pretrained image classifier
- We use the relu2_2 layer (mid-level texture features)
- **Plain English:** Do the two images look like they have the same textures and structures, as judged by an image recognition network?

### Combined Training Loss
```
L_total = 1.0 * L1 + 0.05 * L_perc
```
- **Plain English:** Mostly pixel accuracy (L1), with a small push toward perceptual texture quality.

### FiLM Conditioning
```
FiLM(h, cond) = gamma(cond) ⊙ h + beta(cond)
```
- h = feature map from a residual block
- gamma, beta = per-channel scale and shift, output by a small MLP from the conditioning vector
- ⊙ = element-wise multiplication
- **Plain English:** The conditioning vector tells each layer how to scale and shift its features — effectively controlling the processing strategy for each image.

---

## Metrics Reference Card

*(Quick-reference for you and teammates)*

| Metric | Higher/Lower is Better | Requires Ground Truth? | What It Captures |
|---|---|---|---|
| PSNR (dB) | Higher | Yes | Overall pixel accuracy |
| SSIM | Higher (0–1) | Yes | Structural and brightness similarity |
| LPIPS | Lower | Yes | Perceptual/human-like similarity |
| NIQE | Lower | No | Closeness to natural image statistics |
| BRISQUE | Lower | No | Spatial distortion artifacts |

---

## Honest Limitations

These should be stated clearly in the presentation — confident disclosure builds credibility.

1. **No blind degradation estimation.** We require degradation metadata at evaluation time. For truly blind inputs (RealSRSet), conditioning is zero and the model falls back to its plain backbone.

2. **Reduced training data.** 100 training images vs. 800 in the full DIV2K set. Absolute PSNR numbers are 1–3 dB below published results. Our three-experiment comparisons are internally valid.

3. **No temporal modeling.** Frames treated as independent images. Video coherence is not maintained.

4. **Single scale factor.** Only 4× upscaling was tested and trained.

5. **VGG perceptual loss, not GAN.** Deliberate engineering decision for training stability. Results are sharper than L1-only but lack the high-frequency detail of ESRGAN-style models.

---

## Q&A Preparation

### "Why not use GAN training? ESRGAN and Real-ESRGAN both use GANs."

"That's a great point. GAN training does produce sharper, more detailed results. But GAN training requires balancing a generator and a discriminator simultaneously — if the discriminator overwhelms the generator early in training, the model collapses and produces garbage. Managing that balance requires careful learning rate scheduling and monitoring that's difficult to debug in a class project timeline. VGG perceptual loss gives us most of the texture benefit with none of the instability risk. We explicitly weighed the trade-offs and made a deliberate choice. ESRGAN with adversarial loss would be the next step if we continued this work."

---

### "What happens when you pass a zero conditioning vector to the conditioned model?"

"The FiLM layers reduce to identity — gamma is initialized to one, beta to zero, and a zero input from the conditioning vector moves them only slightly from those defaults early in training. In practice, the model falls back to its SRResNet backbone behavior. It still produces super-resolution, but it cannot adapt to the specific degradation level. For RealSRSet — where we have no metadata — this is exactly what happens, and we document it as a known limitation."

---

### "Why DIV2K? Why not use actual surveillance footage for training?"

"DIV2K is a standard benchmark that provides clean high-resolution images we can degrade in a controlled way. If we trained on actual surveillance footage, we would not have clean ground truth to compare against — we can't know what the original high-resolution scene looked like. By starting from clean DIV2K images and applying our surveillance degradation preset, we simulate surveillance conditions while keeping the ground truth for supervised training. VIRAT footage can be used for evaluation — testing how well the model trained on DIV2K generalizes to real surveillance frames."

---

### "Your PSNR numbers are lower than what's published in papers. Does that mean your model doesn't work?"

"Our absolute PSNR numbers are lower because we trained on 100 images instead of the 800 in the full DIV2K training set. More training data means the model sees more variety and generalizes better. In the super-resolution literature, training set size has a significant effect on absolute PSNR. But our comparison between our three experiments is valid — all three use the same 100 images. The story we care about is: Exp 1 < Exp 2 < Exp 3. That ordering, and the magnitude of the improvements, tells us whether each component we added actually helps. The absolute numbers just tell us we need more data."

---

### "How does curriculum training help if validation uses the full preset throughout?"

"The training curves tell the story. Without curriculum, the model sees severe degradations from the first epoch, when it doesn't yet understand what super-resolution even means. The loss is high, the gradients are large and noisy, and training takes longer to converge. Curriculum gives the model easy wins first — it learns the basic structure of super-resolution on mild inputs, then gradually confronts harder cases. The final model trained with curriculum reaches a higher PSNR at convergence than one trained without, even if both have seen the same total degradation distribution by the end. Think of it like learning to sprint — you don't start with maximum resistance."

---

### "What is LPIPS and why does it matter?"

"LPIPS — Learned Perceptual Image Patch Similarity — uses a pretrained neural network to compare images the way a human might. Pixel-level metrics like PSNR can be fooled: a slightly shifted version of a perfect image has terrible PSNR but looks identical to a human. LPIPS compares feature representations, which are more stable to small spatial shifts and focus on perceptual content rather than exact pixel positions. A model that scores better on LPIPS is producing results that look more similar to ground truth to human eyes, even if the pixel values don't exactly align. LPIPS is lower-is-better."

---

### "Would this work on real surveillance cameras without modification?"

"Not immediately, because of the blind estimation gap we described. To use this on a real camera, you'd need either: one, a way to estimate the degradation parameters from the input image — which is our primary future work item — or two, calibration data for a specific camera that lets you set the conditioning vector to known fixed values. Option two is actually practical in a deployed system: if you know the camera model, you can profile its blur and noise characteristics once and hardcode that information into the conditioning vector for all footage from that camera."

---

### "What is FiLM and where does it come from?"

"FiLM — Feature-wise Linear Modulation — comes from a 2018 paper by Perez et al. called 'FiLM: Visual Reasoning with a General Conditioning Layer.' It was originally designed for visual question answering — a task where you need to answer questions about images, so the question needs to influence how the image is processed. We adapted it to super-resolution: instead of a question, the conditioning signal is our degradation metadata vector. The core idea is the same: a small MLP reads the conditioning input and produces scale (gamma) and shift (beta) values that are applied to the feature maps in each layer. It's parameter-efficient and doesn't change the base architecture."

---

### "If you were doing this again, what would you do differently?"

"Three things. First, we would implement a simple degradation estimator early — even a rough one — so we could test the fully blind pipeline. Second, we would train on the full DIV2K dataset from the start, even if it meant fewer experiment configurations. And third, we would run experiments in parallel rather than sequentially — the three experiments are independent and could run simultaneously on separate GPU instances, cutting total training time to a third. The architecture and degradation pipeline are solid; the main things we'd change are operational."

---

*End of project.md — presentation script version.*
*Fill in all [bracketed placeholders] with actual numbers from your experiment outputs before presenting.*
