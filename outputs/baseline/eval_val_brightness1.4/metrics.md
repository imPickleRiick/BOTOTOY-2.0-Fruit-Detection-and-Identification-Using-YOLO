# OpenCV baseline - evaluation on `val` (686 images, brightness x1.4)

| Class | GT boxes | Predictions | Precision@0.5 | Recall@0.5 | F1@0.5 | AP50 | AP50-95 |
|---|---|---|---|---|---|---|---|
| Apple | 448 | 158 | 0.240 | 0.085 | 0.125 | 0.138 | 0.091 |
| Banana | 284 | 655 | 0.061 | 0.141 | 0.085 | 0.044 | 0.023 |
| Grape | 572 | 430 | 0.135 | 0.101 | 0.116 | 0.082 | 0.045 |
| Orange | 883 | 380 | 0.082 | 0.035 | 0.049 | 0.051 | 0.033 |
| Pineapple | 140 | 166 | 0.012 | 0.014 | 0.013 | 0.006 | 0.001 |
| Watermelon | 164 | 533 | 0.034 | 0.110 | 0.052 | 0.044 | 0.022 |
| **Mean** | | | **0.094** | **0.081** | **0.073** | **0.061** | **0.036** |

- Localization recall (any fruit found at IoU 0.5, label ignored): **0.152**
- Count error per image: mean absolute 3.27, mean signed -0.246 (negative = under-count), exact count 15.3%
- Time per image: detector 56.5 ms (median 57.0), end-to-end 65.6 ms (AMD Ryzen 5 5600U with Radeon Graphics)

Example images: `examples/` (left = ground truth, right = baseline prediction).
