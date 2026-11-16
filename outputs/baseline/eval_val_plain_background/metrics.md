# OpenCV baseline - evaluation on `val` (76 images, plain background only, brightness x1.0)

| Class | GT boxes | Predictions | Precision@0.5 | Recall@0.5 | F1@0.5 | AP50 | AP50-95 |
|---|---|---|---|---|---|---|---|
| Apple | 57 | 17 | 0.353 | 0.105 | 0.162 | 0.209 | 0.164 |
| Banana | 41 | 46 | 0.130 | 0.146 | 0.138 | 0.091 | 0.050 |
| Grape | 18 | 13 | 0.077 | 0.056 | 0.065 | 0.055 | 0.049 |
| Orange | 25 | 27 | 0.222 | 0.240 | 0.231 | 0.257 | 0.179 |
| Pineapple | 5 | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Watermelon | 12 | 33 | 0.061 | 0.167 | 0.089 | 0.037 | 0.012 |
| **Mean** | | | **0.141** | **0.119** | **0.114** | **0.108** | **0.076** |

- Localization recall (any fruit found at IoU 0.5, label ignored): **0.392**
- Count error per image: mean absolute 1.316, mean signed -0.237 (negative = under-count), exact count 38.2%
- Time per image: detector 41.4 ms (median 39.4), end-to-end 51.0 ms (AMD Ryzen 5 5600U with Radeon Graphics)

Example images: `examples/` (left = ground truth, right = baseline prediction).
