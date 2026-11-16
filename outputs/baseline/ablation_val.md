# OpenCV baseline - improvement log (validation split, 686 images)

Each step keeps all previous changes. Metrics use IoU 0.5 and the same AP formula as YOLO.

| Step | Why | Predictions | Precision | Recall | mAP50 | mAP50-95 | Localization recall | Count MAE | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| A. Initial rules | color masks + shape rules, every region >= 0.2% of the image | 6658 | 0.035 | 0.090 | 0.026 | 0.015 | 0.215 | 7.608 | 84.6 |
| B. Minimum region size 1% | regions under 1% of the image were ~98% false positives | 2899 | 0.075 | 0.074 | 0.047 | 0.029 | 0.171 | 3.656 | 58.1 |
| C. Reject irregular green regions | green 'grape' regions were mostly leaves/grass (0.4% correct) | 2337 | 0.084 | 0.072 | 0.052 | 0.032 | 0.157 | 3.324 | 56.2 |
| D. Size-aware rule score | rank larger regions higher: they are far more often real fruits | 2337 | 0.084 | 0.072 | 0.053 | 0.032 | 0.157 | 3.324 | 56.1 |
| E. Watershed split of touching fruits | split blobs of touching round fruits (warm/red) | 2815 | 0.074 | 0.075 | 0.049 | 0.029 | 0.167 | 3.802 | 101.5 |
