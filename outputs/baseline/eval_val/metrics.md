# OpenCV baseline - evaluation on `val` (686 images, brightness x1.0)

| Class | GT boxes | Predictions | Precision@0.5 | Recall@0.5 | F1@0.5 | AP50 | AP50-95 |
|---|---|---|---|---|---|---|---|
| Apple | 448 | 165 | 0.200 | 0.074 | 0.108 | 0.115 | 0.076 |
| Banana | 284 | 580 | 0.060 | 0.123 | 0.081 | 0.041 | 0.021 |
| Grape | 572 | 480 | 0.117 | 0.098 | 0.106 | 0.071 | 0.042 |
| Orange | 883 | 445 | 0.101 | 0.051 | 0.068 | 0.064 | 0.040 |
| Pineapple | 140 | 144 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Watermelon | 164 | 523 | 0.027 | 0.085 | 0.041 | 0.030 | 0.013 |
| **Mean** | | | **0.084** | **0.072** | **0.067** | **0.053** | **0.032** |

- Localization recall (any fruit found at IoU 0.5, label ignored): **0.157**
- Count error per image: mean absolute 3.324, mean signed -0.224 (negative = under-count), exact count 14.7%
- Time per image: detector 57.7 ms (median 56.4), end-to-end 66.3 ms (AMD Ryzen 5 5600U with Radeon Graphics)

Example images: `examples/` (left = ground truth, right = baseline prediction).
