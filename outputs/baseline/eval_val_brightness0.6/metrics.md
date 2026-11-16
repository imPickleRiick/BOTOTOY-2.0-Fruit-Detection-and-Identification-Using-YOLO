# OpenCV baseline - evaluation on `val` (686 images, brightness x0.6)

| Class | GT boxes | Predictions | Precision@0.5 | Recall@0.5 | F1@0.5 | AP50 | AP50-95 |
|---|---|---|---|---|---|---|---|
| Apple | 448 | 159 | 0.182 | 0.065 | 0.096 | 0.101 | 0.062 |
| Banana | 284 | 579 | 0.074 | 0.151 | 0.100 | 0.054 | 0.030 |
| Grape | 572 | 467 | 0.105 | 0.086 | 0.094 | 0.062 | 0.037 |
| Orange | 883 | 407 | 0.125 | 0.058 | 0.079 | 0.080 | 0.050 |
| Pineapple | 140 | 128 | 0.008 | 0.007 | 0.007 | 0.004 | 0.002 |
| Watermelon | 164 | 474 | 0.036 | 0.104 | 0.053 | 0.042 | 0.015 |
| **Mean** | | | **0.088** | **0.078** | **0.072** | **0.057** | **0.033** |

- Localization recall (any fruit found at IoU 0.5, label ignored): **0.155**
- Count error per image: mean absolute 3.322, mean signed -0.404 (negative = under-count), exact count 15.9%
- Time per image: detector 57.8 ms (median 56.9), end-to-end 68.3 ms (AMD Ryzen 5 5600U with Radeon Graphics)

Example images: `examples/` (left = ground truth, right = baseline prediction).
