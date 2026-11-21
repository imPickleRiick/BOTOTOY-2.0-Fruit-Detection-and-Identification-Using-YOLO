# Fruit Reader (v2): presentation and defense guide

For the team, before the final presentation. It follows the rubric (technical implementation 30%, computer
vision concepts 25%, problem-solving 25%, presentation and defense 20%). Every number here comes from the
report and the result files; if the guide and the code ever disagree, the code is right.

**Read all of it, not only your own part.** The panel can ask any member about any part.

---

## 1. The 60-second pitch (practice saying it aloud)

> About 332,000 Filipinos are blind and 2.18 million have low vision (DOH, 2017). At the palengke, fruit is sold
> loose and without labels, and many fruits feel alike. Our first version drew boxes around six fruits on a
> screen, which helps no one who cannot see. So we turned it into the **Fruit Reader**: point a phone at a fruit
> and it **says** what it is, in English and Filipino, how many, and where, for 31 fruits common in the
> Philippines. It first checks the photo with classic computer vision and tells the user if it is too dark or
> blurry. Then YOLO11n, a deep-learning detector we retrained on 17,123 images, finds the fruit. It names the
> right fruit in 92.7% of test photos, in 61 milliseconds per photo on a laptop. Its biggest open issue is
> honest: we have not yet tested it with blind users.

## 2. Suggested flow (about 10 minutes)

Adjust the split to who actually did what (report §11).

| Time | Who | What | Show |
|---|---|---|---|
| 1 min | anyone | Pitch (section 1) | Fruit Reader start screen |
| 1.5 min | James | Data: two datasets, leakage removal, 31 classes | Dashboard → Training data |
| 1.5 min | Kyle | Classic CV: the photo-quality check | Photo mode with a dark photo |
| 2 min | Ian | YOLO11n, training, the sentence, live reader | Architecture figure (report §5) |
| 2 min | all | **Live demo on the phone** | Phone (screen mirrored if possible) |
| 1.5 min | anyone | Results and limits | Dashboard → Results and per-fruit chart; About → Limits |

## 3. Demo plan

**Before the class (do all of these):**
1. Charge the laptop and the phone. Turn on the phone's **hotspot** and connect the laptop to it (school Wi-Fi
   usually blocks phone-to-laptop connections).
2. Start the app: `python app.py --https`. Open the printed `https://…:9443` address on the phone, tap
   **Advanced → Proceed** on the warning, allow the camera and tap **Start** once to check it.
3. Turn the phone volume up. Test that it speaks.
4. Bring fruits that read well: **mango, banana, orange, pineapple, apple**, and a grape bunch if possible.
   Avoid papaya (weakest, 76%) and green citrus (it was called papaya once).
5. Have the **backup video** ready on the laptop and the phone.

**During the demo:**
1. Live mode: point at one fruit, hold still about 2–3 seconds, let it speak.
2. Two different fruits side by side: it says both, with left / middle / right.
3. Point at an empty, dark spot (or cover most of the lens with a hand) for a few seconds: with no fruit in
   view and a bad picture, it says "too dark" (or "blurry") and what to do. While a fruit is found, it names the
   fruit instead.
4. Optional: Photo mode with a dark photo to show the warning plus the answer.

**If something fails:**

| Problem | Fix |
|---|---|
| Phone cannot open the page | Laptop and phone on the same hotspot? Restart the app after changing networks (new address). |
| "Connection is not private" | Expected: **Advanced → Proceed** (self-signed certificate). |
| Camera does not start | Must be the **https** address; allow camera in the browser's site settings; close other camera apps. |
| No sound | Volume up; tap **Repeat**; some phones need one tap before the first speech. |
| Wrong fruit | Say so calmly. It is in our limits (section 6.6). Point again with better light and a plain background. |
| Nothing works | Play the backup video and continue. |

## 4. Rubric map: what proves each criterion

| Criterion | What to show or say |
|---|---|
| 1. Technical implementation (30%) | Live demo; 61 ms per photo on a CPU; 101 automated tests pass; works on a phone with no app install |
| 2. Computer vision concepts (25%) | Classic CV: Laplacian sharpness, directional second derivatives for motion blur, brightness, glare. Deep learning: CNN detector, transfer learning, IoU, NMS, confidence, mAP. Evaluation: locked test set, leakage removal (dHash) |
| 3. Problem-solving and innovation (25%) | A real user group with real numbers; output by voice, not by screen; Filipino names; spoken photo advice; "speak when 3 frames agree"; phone camera via HTTPS; honest limits and a plan |
| 4. Presentation and defense (20%) | The pitch, a clean demo, and the answers below |

## 5. Numbers to remember

| What | Value |
|---|---|
| Fruits | **31** (19 with Filipino names) |
| Data | **17,123** images, 85,228 boxes, two public datasets; split 13,437 / 2,254 / 1,432 |
| Leakage removed | 393 images from the Roboflow set + 1 cross-dataset |
| Model | YOLO11n, 2.6 million parameters, 5.5 MB, COCO-pretrained |
| Training | Colab T4 GPU, 125 min, stopped at epoch 45 of 50 (quota); best = epoch 44 |
| Test mAP@0.50 / @0.50:0.95 | **0.826 / 0.652** (1,432 test images) |
| Reader names ≥ 1 fruit right | **92.7%** of test photos; exactly right set **87.0%** |
| Hard conditions | **103/132** (v1: 100/132); motion blur 9/12 (v1: 4/12) |
| Speed | **61 ms** per photo on the laptop CPU; quality check 16 ms; about 1.5 frames per second live |
| Live | speaks after **3** matching frames (about 2–3 s), repeats after 15 s |
| Needless photo warnings | **24.4%** of normal test photos |
| Version 1, for comparison | 6 fruits, test mAP@0.50 0.490 |

## 6. Likely questions and short answers

### 6.1 Problem and design choices

**Why not the banana ripeness advisor our instructor suggested?**
We looked for data first. No public dataset has banana variety **and** ripeness labels, and labeling our own
would not fit three days; a model trained without such labels would only guess. The reader reused our tested
pipeline and solves a problem with a clear user. Ripeness for selected fruits is in our future work.

**Who exactly is the user, and how do they use it without seeing?**
Blind and low-vision people. They never need to read the screen: the answer is spoken automatically, the
buttons are large, and the text is also announced by screen readers (TalkBack, VoiceOver). In live mode they
do not even press a button to take a photo.

**Why does it need a laptop? Why not only the phone?**
The model runs in Python. Exporting YOLO11n to a phone format (TFLite or ONNX) is possible but needs app
development we did not have time for. For now the phone is the camera, screen and speaker. It is our first
technical next step.

**Is the user's camera data sent to the internet?**
No. Frames go from the phone to the laptop over the local hotspot or Wi-Fi, are processed there, and are not
saved or uploaded to a cloud service. The speech is made by the phone's own browser.

**Why say "many" instead of the exact count above five?**
Counting fruit in piles is unreliable (our version 1 error review showed the model sometimes boxes a whole pile
as one). Saying "many" is more honest than a wrong number.

**Why does it say "probably" or "I am not sure"?**
So the user knows when to double-check by touch or smell. "I see" for confidence ≥ 0.6, "probably" for
0.4–0.6, "not sure" below 0.4.

### 6.2 Classic computer vision (the photo-quality check)

**How do you detect a blurry photo?**
With the **variance of the Laplacian**. The Laplacian is a second-derivative filter; it responds strongly at
edges. A sharp photo has many strong edges, so the values vary a lot; a blurred photo has soft edges, so the
variance is low. Below 60 we say "blurry".

**How do you tell motion blur from normal blur?**
A shaking camera smears detail in **one direction**. We filter the image with second-derivative kernels in four
directions (0°, 45°, 90°, 135°) and compare the weakest to the strongest energy. If one direction has almost no
detail left (ratio below 0.15), it is motion blur.

**And dark or too bright?**
Mean gray level below 60 → too dark; above 190, or more than 20% pure-white pixels → too bright.

**How did you choose those thresholds?**
By measuring the 12 unseen photos and 8 demo photos and their simulated dark and blurred versions, never the
test set. Every dark, blurred and low-resolution version was flagged, and no original photo was.

**Does it work well?**
It catches bad photos, but it also warns on 24% of normal test photos, mostly product photos on white
backgrounds ("too bright") and smooth close-ups ("blurry"). A warning never blocks the answer: the fruit is
still named after the advice. Tuning it on real phone photos is our next step.

### 6.3 Deep learning (YOLO11n)

**What is YOLO?**
"You Only Look Once": a one-stage object detector. One pass of a convolutional neural network predicts, for
every region of the image at three scales, a box, a class and a confidence. There is no separate
region-proposal step, so it is fast.

**What is transfer learning and why use it?**
We start from weights already trained on COCO (80 everyday object classes), so the early layers already
recognize edges, textures and shapes. Ultralytics replaces the last layer with a 31-class head and we fine-tune
all layers. This needs far less data and time than training from zero.

**Why the smallest model, YOLO11n?**
Our laptops have no GPU. The nano model runs in 61 ms per photo on a CPU, fast enough for live reading.

**What is IoU? What is NMS?**
IoU (intersection over union) is the overlap of two boxes divided by their combined area, 0 to 1. The model
often predicts several boxes for one fruit; **non-maximum suppression** keeps the most confident one and removes
others of the same class that overlap it with IoU above 0.7.

**What is the confidence threshold?**
Boxes with confidence below 0.25 are discarded. Lower finds more fruit but adds false alarms. The reader lets
the user change it in Settings.

**What do mAP@0.50 and mAP@0.50:0.95 mean?**
For each fruit, **AP** is the area under the precision–recall curve: how precise the model stays as it finds
more of the fruits. A prediction counts as correct if it has the right class and IoU ≥ 0.5 with a labeled
box. **mAP@0.50** averages AP over the 31 fruits. **mAP@0.50:0.95** averages it over stricter IoU thresholds
(0.50 to 0.95), so it also rewards precise boxes.

**Precision vs. recall?**
Precision: of the fruits we reported, how many were real. Recall: of the real fruits, how many we found.

### 6.4 Data and training

**Where does the data come from?**
Two public datasets: "Fruit" by trucly on Roboflow (CC BY 4.0, 30 tropical and common fruits) and the version 1
Fruit Detection dataset (CC0) for apple and grape. 17,123 images in total.

**What is data leakage, and how did you prevent it?**
Leakage is when the same photo (or a near copy) is in both training and testing, which makes the score look
better than it really is. Roboflow names copies after their source photo, so we removed test and validation
images whose source photo was in another split. We also compared every image with a **dHash** (a 64-bit
fingerprint of the image's brightness pattern) and removed near-copies across splits: 393 images in total, plus
1 between the two datasets.

**Why merge tangerines into orange?**
They look very alike, and a blind user needs "orange (dalandan)", not the difference between two citrus types.

**Training stopped at epoch 45 of 50. Is the model incomplete?**
No. The free Colab GPU quota ran out. Ultralytics saves the best checkpoint as training goes, so we used the
one from epoch 44. Validation mAP@0.50 had stayed at about 0.82–0.83 since epoch 37, so the missing epochs would have changed
little.

**Why was the test set used only once?**
If we kept checking the test set while tuning, we would slowly fit the test set itself, and the score would no
longer show how the model does on new photos. All tuning used the validation set.

### 6.5 Results

**Version 1 scored 0.490, version 2 scores 0.826. Is the new model that much better?**
Partly, but the test sets differ, so the numbers are not a direct comparison. On the **same** 681 version 1
test photos, the new reader names apple, banana, grape, orange, pineapple and watermelon as often as version 1
did, and is right more often when it names one, while also knowing 25 more fruits. On hard photos it reads
103/132 vs 100/132, with motion blur much better (9 vs 4).

**Why is the apple and grape box recall so low (about 0.28)?**
Their test photos all come from the version 1 dataset, which labels crowded scenes inconsistently: one box per
fruit in some photos, one box per bunch or pile in others. Box scores punish that. But the reader still **names**
apple in 85% and grape in 93% of the photos that have them, and naming is what the user hears.

**Why measure "named correctly" instead of only mAP?**
A blind user does not see boxes; they hear names. So we score the set of names the reader says in each photo:
at least one right in 92.7%, exactly right in 87.0%.

**Which fruits are weakest?**
Papaya (named in 76% of its photos) and rambutan (82%). Avocado and guava are named when they are there but are
also said when they are not (right only 46% and 60% of the time; few test photos).

### 6.6 Implementation

**How does the live reader decide when to speak?**
The phone sends about 1.5 frames per second. The reader speaks when the **same set of fruits** appears in **3
frames in a row** (about 2–3 seconds), so it does not shout every guess. It does not repeat the same answer for
15 seconds, and it resets after 3 frames without fruit.

**Why HTTPS and the security warning?**
Browsers only allow the camera on secure (https) pages. We create our own certificate for the laptop, which the
phone does not know, so it warns once. A small proxy we wrote adds HTTPS in front of the app.

**How is the sentence spoken?**
By the Web Speech API in the phone's browser, with an English voice; Filipino names are said by that voice too,
so they can sound unnatural.

**How did you test it?**
101 automated tests (sentences, names, quality check, QR code, HTTPS proxy, live announcer, app pages), the
locked test set, the reader test (naming, hard conditions, false alarms) and live tests on a team phone.

### 6.7 Limits and next steps

**What are the main limitations?**
1. Not tested with blind or low-vision users yet.
2. 24% needless photo warnings.
3. Unusual colors fool it (a green orange was called papaya, a yellow watermelon "probably a guava").
4. Needs a laptop and a shared network.
5. 31 fruits only: no calamansi, atis, santol or langka yet.
6. It does not judge ripeness or freshness.

**What would you do next?**
Test with blind users (for example a school or association for the blind); tune the photo check on real phone
photos; run the model on the phone itself; add more local fruits; add ripeness for selected fruits, as our
instructor suggested.

## 7. Optional: a blindfold test before the presentation

A quick, honest stand-in for user testing. Call it a **blindfold test with sighted classmates**, never a test
with blind users.

1. 3–5 classmates, each blindfolded after a 1-minute explanation ("tap Start, point, hold still").
2. Each gets 5 fruits one at a time, in a random order, from the reliable list in section 3.
3. Record per fruit: what the reader said, whether it was right, and the seconds from pointing to speech
   (a teammate with a stopwatch).
4. Ask each person afterwards: "How clear was the sentence? (1–5)" and "What would you change?"

| Person | Fruit | Reader said | Right? | Seconds | Clarity (1–5) | Comment |
|---|---|---|---|---|---|---|
| | | | | | | |

Report the share of correct readings, the median time, the average clarity and the most common comment, on one
slide or one paragraph. Even a small result like "23 of 25 fruits named correctly, median 3 s" is stronger
evidence than none.
