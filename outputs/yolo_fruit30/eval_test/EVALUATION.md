# YOLO11n evaluation - `test` split (751 images)

Weights: `bototoy_fruit30_yolo11n.pt` (sha256 `cc078f9fc13c…`)  
Hardware: AMD Ryzen 5 5600U with Radeon Graphics, 5.9 GB RAM, CPU inference

## Ultralytics validation metrics

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|
| ambarella_fruit | 0.840 | 0.706 | 0.865 | 0.753 |
| avocado | 0.683 | 0.912 | 0.888 | 0.727 |
| banana | 0.841 | 0.770 | 0.896 | 0.688 |
| burmese_grape | 0.692 | 0.486 | 0.571 | 0.450 |
| carambola | 0.914 | 0.917 | 0.941 | 0.747 |
| cempedak | 0.863 | 0.879 | 0.939 | 0.725 |
| coconut | 0.652 | 0.680 | 0.648 | 0.533 |
| custard_apple | 0.919 | 0.908 | 0.948 | 0.724 |
| dragon_fruit | 0.918 | 0.876 | 0.959 | 0.618 |
| durian | 0.873 | 0.872 | 0.945 | 0.672 |
| grapefruit | 0.811 | 0.923 | 0.949 | 0.819 |
| guava | 0.817 | 0.900 | 0.954 | 0.831 |
| lanzones | 0.909 | 0.692 | 0.873 | 0.766 |
| lychee | 0.614 | 0.654 | 0.617 | 0.468 |
| mango | 0.930 | 0.908 | 0.971 | 0.785 |
| mangosteen | 0.927 | 0.963 | 0.981 | 0.785 |
| melon | 0.918 | 0.959 | 0.983 | 0.901 |
| orange | 0.870 | 0.846 | 0.939 | 0.758 |
| papayas | 0.917 | 0.722 | 0.892 | 0.747 |
| pear | 0.925 | 0.936 | 0.973 | 0.861 |
| persimmon | 0.947 | 0.974 | 0.991 | 0.898 |
| pineapple | 0.900 | 0.826 | 0.925 | 0.571 |
| pomegranate | 0.964 | 0.860 | 0.935 | 0.690 |
| rambutan | 0.759 | 0.644 | 0.770 | 0.538 |
| rose_apple | 0.884 | 0.850 | 0.937 | 0.716 |
| sapodilla | 0.959 | 0.934 | 0.965 | 0.855 |
| star_apple | 0.942 | 0.891 | 0.951 | 0.700 |
| tangerines | 0.860 | 0.873 | 0.908 | 0.721 |
| watermelon | 0.848 | 0.908 | 0.915 | 0.766 |
| wood_apple | 0.652 | 1.000 | 0.746 | 0.634 |
| **All** | **0.852** | **0.842** | **0.892** | **0.715** |

_precision/recall are at the confidence that maximizes mean F1 (Ultralytics convention)._ Plots: `ultralytics/` (confusion matrix, PR / F1 / P / R curves).

## Proposal targets

| Target | Result | Met? |
|---|---|---|
| mAP50 >= 0.80 | 0.8925 | ✅ yes |
| mAP50-95 >= 0.50 | 0.7149 | ✅ yes |
| recall >= 0.70 for every class | ambarella_fruit 0.71, avocado 0.91, banana 0.77, burmese_grape 0.49, carambola 0.92, cempedak 0.88, coconut 0.68, custard_apple 0.91, dragon_fruit 0.88, durian 0.87, grapefruit 0.92, guava 0.90, lanzones 0.69, lychee 0.65, mango 0.91, mangosteen 0.96, melon 0.96, orange 0.85, papayas 0.72, pear 0.94, persimmon 0.97, pineapple 0.83, pomegranate 0.86, rambutan 0.64, rose_apple 0.85, sapodilla 0.93, star_apple 0.89, tangerines 0.87, watermelon 0.91, wood_apple 1.00 | ❌ no |
| end-to-end <= 1 s per image | 0.067 | ✅ yes |

## At the app's operating point (confidence ≥ 0.25, NMS IoU 0.7)

Computed with the same code as the OpenCV baseline (`src/utils/metrics.py`).

| Class | GT boxes | Predictions | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---|---|---|---|---|
| ambarella_fruit | 68 | 61 | 0.803 | 0.721 | 0.760 |
| avocado | 34 | 52 | 0.596 | 0.912 | 0.721 |
| banana | 61 | 74 | 0.730 | 0.885 | 0.800 |
| burmese_grape | 183 | 164 | 0.598 | 0.535 | 0.565 |
| carambola | 72 | 84 | 0.821 | 0.958 | 0.885 |
| cempedak | 83 | 113 | 0.699 | 0.952 | 0.806 |
| coconut | 55 | 73 | 0.548 | 0.727 | 0.625 |
| custard_apple | 76 | 75 | 0.933 | 0.921 | 0.927 |
| dragon_fruit | 218 | 251 | 0.825 | 0.950 | 0.883 |
| durian | 141 | 175 | 0.743 | 0.922 | 0.823 |
| grapefruit | 91 | 114 | 0.737 | 0.923 | 0.820 |
| guava | 60 | 77 | 0.740 | 0.950 | 0.832 |
| lanzones | 94 | 90 | 0.767 | 0.734 | 0.750 |
| lychee | 226 | 259 | 0.595 | 0.681 | 0.635 |
| mango | 750 | 846 | 0.861 | 0.971 | 0.912 |
| mangosteen | 80 | 91 | 0.846 | 0.963 | 0.901 |
| melon | 49 | 53 | 0.906 | 0.980 | 0.941 |
| orange | 201 | 261 | 0.701 | 0.910 | 0.792 |
| papayas | 107 | 98 | 0.827 | 0.757 | 0.790 |
| pear | 47 | 49 | 0.939 | 0.979 | 0.958 |
| persimmon | 117 | 134 | 0.866 | 0.992 | 0.924 |
| pineapple | 328 | 413 | 0.731 | 0.921 | 0.815 |
| pomegranate | 31 | 29 | 0.931 | 0.871 | 0.900 |
| rambutan | 174 | 164 | 0.616 | 0.581 | 0.598 |
| rose_apple | 295 | 321 | 0.819 | 0.891 | 0.854 |
| sapodilla | 91 | 91 | 0.912 | 0.912 | 0.912 |
| star_apple | 266 | 279 | 0.857 | 0.898 | 0.877 |
| tangerines | 251 | 301 | 0.754 | 0.904 | 0.823 |
| watermelon | 221 | 300 | 0.717 | 0.973 | 0.825 |
| wood_apple | 19 | 31 | 0.613 | 1.000 | 0.760 |
| **Mean** | | | **0.768** | **0.876** | **0.814** |

- AP with our implementation (all predictions): mAP50 **0.877**, mAP50-95 **0.707**
- Localization recall (label ignored): 0.899
- Counting: mean absolute error 1.561 fruits/image, mean signed 0.844, exact count 50.5%

## Processing time (this computer, CPU)

- Model stages (mean ms): preprocess 1.89, inference 54.32, postprocess/NMS 1.34
- End-to-end per image: mean **66.7 ms**, median 62.2 ms, 95th percentile 104.4 ms
- end-to-end = read image + preprocess + inference + NMS + draw boxes, batch size 1, CPU, after one warm-up image

## Error analysis

`examples/`: `tp_*` = every fruit found with no false alarms; `fp_*` = images with the most false positives (red boxes); `fn_*` = images with the most missed fruits (magenta boxes).
 Saved: {'tp': 6, 'fp': 6, 'fn': 6}.
