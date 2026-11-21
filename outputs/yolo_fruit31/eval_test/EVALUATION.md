# YOLO11n evaluation - `test` split (1432 images)

Weights: `bototoy_fruit31_yolo11n.pt` (sha256 `57a494b3f0af…`)  
Hardware: AMD Ryzen 5 5600U with Radeon Graphics, 5.9 GB RAM, CPU inference

## Ultralytics validation metrics

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|
| ambarella_fruit | 0.708 | 0.779 | 0.779 | 0.693 |
| avocado | 0.795 | 0.941 | 0.927 | 0.760 |
| banana | 0.768 | 0.490 | 0.576 | 0.362 |
| burmese_grape | 0.646 | 0.470 | 0.520 | 0.409 |
| carambola | 0.837 | 0.925 | 0.940 | 0.738 |
| cempedak | 0.822 | 0.916 | 0.956 | 0.730 |
| coconut | 0.638 | 0.705 | 0.726 | 0.586 |
| custard_apple | 0.854 | 0.920 | 0.931 | 0.725 |
| dragon_fruit | 0.894 | 0.844 | 0.941 | 0.607 |
| durian | 0.847 | 0.879 | 0.931 | 0.672 |
| grapefruit | 0.767 | 0.912 | 0.906 | 0.764 |
| guava | 0.823 | 0.917 | 0.942 | 0.800 |
| lanzones | 0.789 | 0.702 | 0.787 | 0.684 |
| lychee | 0.677 | 0.678 | 0.668 | 0.512 |
| mango | 0.911 | 0.904 | 0.964 | 0.774 |
| mangosteen | 0.906 | 0.925 | 0.970 | 0.744 |
| melon | 0.735 | 0.963 | 0.952 | 0.871 |
| orange | 0.771 | 0.549 | 0.672 | 0.471 |
| papayas | 0.870 | 0.626 | 0.830 | 0.685 |
| pear | 0.951 | 0.957 | 0.984 | 0.850 |
| persimmon | 0.949 | 0.962 | 0.987 | 0.888 |
| pineapple | 0.831 | 0.723 | 0.826 | 0.514 |
| pomegranate | 0.605 | 0.871 | 0.781 | 0.598 |
| rambutan | 0.697 | 0.695 | 0.766 | 0.545 |
| rose_apple | 0.831 | 0.881 | 0.930 | 0.705 |
| sapodilla | 0.901 | 0.956 | 0.969 | 0.845 |
| star_apple | 0.920 | 0.863 | 0.926 | 0.676 |
| watermelon | 0.790 | 0.786 | 0.829 | 0.674 |
| wood_apple | 0.651 | 0.947 | 0.877 | 0.750 |
| apple | 0.763 | 0.277 | 0.427 | 0.306 |
| grape | 0.806 | 0.289 | 0.398 | 0.269 |
| **All** | **0.798** | **0.782** | **0.826** | **0.652** |

_precision/recall are at the confidence that maximizes mean F1 (Ultralytics convention)._ Plots: `ultralytics/` (confusion matrix, PR / F1 / P / R curves).

## Proposal targets

| Target | Result | Met? |
|---|---|---|
| mAP50 >= 0.80 | 0.8263 | ✅ yes |
| mAP50-95 >= 0.50 | 0.6518 | ✅ yes |
| recall >= 0.70 for every class | ambarella_fruit 0.78, avocado 0.94, banana 0.49, burmese_grape 0.47, carambola 0.93, cempedak 0.92, coconut 0.70, custard_apple 0.92, dragon_fruit 0.84, durian 0.88, grapefruit 0.91, guava 0.92, lanzones 0.70, lychee 0.68, mango 0.90, mangosteen 0.93, melon 0.96, orange 0.55, papayas 0.63, pear 0.96, persimmon 0.96, pineapple 0.72, pomegranate 0.87, rambutan 0.70, rose_apple 0.88, sapodilla 0.96, star_apple 0.86, watermelon 0.79, wood_apple 0.95, apple 0.28, grape 0.29 | ❌ no |
| end-to-end <= 1 s per image | 0.061 | ✅ yes |

## At the app's operating point (confidence ≥ 0.25, NMS IoU 0.7)

Computed with the same code as the OpenCV baseline (`src/utils/metrics.py`).

| Class | GT boxes | Predictions | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---|---|---|---|---|
| ambarella_fruit | 68 | 76 | 0.658 | 0.735 | 0.694 |
| avocado | 34 | 52 | 0.615 | 0.941 | 0.744 |
| banana | 338 | 255 | 0.690 | 0.521 | 0.594 |
| burmese_grape | 183 | 131 | 0.656 | 0.470 | 0.548 |
| carambola | 72 | 82 | 0.817 | 0.931 | 0.870 |
| cempedak | 83 | 122 | 0.656 | 0.964 | 0.780 |
| coconut | 55 | 73 | 0.548 | 0.727 | 0.625 |
| custard_apple | 76 | 84 | 0.833 | 0.921 | 0.875 |
| dragon_fruit | 218 | 250 | 0.808 | 0.927 | 0.863 |
| durian | 141 | 177 | 0.729 | 0.915 | 0.811 |
| grapefruit | 91 | 117 | 0.709 | 0.912 | 0.798 |
| guava | 60 | 85 | 0.682 | 0.967 | 0.800 |
| lanzones | 94 | 99 | 0.667 | 0.702 | 0.684 |
| lychee | 226 | 256 | 0.586 | 0.664 | 0.622 |
| mango | 750 | 832 | 0.859 | 0.953 | 0.904 |
| mangosteen | 80 | 96 | 0.823 | 0.988 | 0.898 |
| melon | 49 | 62 | 0.774 | 0.980 | 0.865 |
| orange | 1179 | 1079 | 0.660 | 0.604 | 0.631 |
| papayas | 107 | 89 | 0.820 | 0.682 | 0.745 |
| pear | 47 | 47 | 0.957 | 0.957 | 0.957 |
| persimmon | 117 | 129 | 0.899 | 0.992 | 0.943 |
| pineapple | 429 | 460 | 0.713 | 0.765 | 0.738 |
| pomegranate | 31 | 44 | 0.636 | 0.903 | 0.747 |
| rambutan | 174 | 194 | 0.624 | 0.695 | 0.658 |
| rose_apple | 295 | 357 | 0.759 | 0.919 | 0.831 |
| sapodilla | 91 | 104 | 0.837 | 0.956 | 0.892 |
| star_apple | 266 | 279 | 0.853 | 0.895 | 0.873 |
| watermelon | 350 | 416 | 0.709 | 0.843 | 0.770 |
| wood_apple | 19 | 31 | 0.613 | 1.000 | 0.760 |
| apple | 531 | 260 | 0.658 | 0.322 | 0.432 |
| grape | 603 | 301 | 0.678 | 0.338 | 0.451 |
| **Mean** | | | **0.727** | **0.809** | **0.755** |

- AP with our implementation (all predictions): mAP50 **0.807**, mAP50-95 **0.641**
- Localization recall (label ignored): 0.727
- Counting: mean absolute error 1.82 fruits/image, mean signed -0.152, exact count 48.2%

## Processing time (this computer, CPU)

- Model stages (mean ms): preprocess 1.74, inference 50.47, postprocess/NMS 1.23
- End-to-end per image: mean **60.9 ms**, median 58.7 ms, 95th percentile 84.5 ms
- end-to-end = read image + preprocess + inference + NMS + draw boxes, batch size 1, CPU, after one warm-up image

## Error analysis

`examples/`: `tp_*` = every fruit found with no false alarms; `fp_*` = images with the most false positives (red boxes); `fn_*` = images with the most missed fruits (magenta boxes).
 Saved: {'tp': 6, 'fp': 6, 'fn': 6}.
