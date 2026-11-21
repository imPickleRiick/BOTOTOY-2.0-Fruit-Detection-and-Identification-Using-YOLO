# Training the combined 31-fruit model (v2) on Google Colab

Time needed: about 10 minutes of setup + about 2 hours of training. Use a laptop or PC browser (not a phone).

## 1. Put the dataset on Google Drive (once)

1. Open [drive.google.com](https://drive.google.com) with the Google account you will use in Colab.
2. Create a folder named **`BOTOTOY_2.0`** in **My Drive** (it may already exist from the v1 training).
3. Upload **`Fruit.v1i.yolov8.zip`** (745 MB, from your Downloads folder) into that folder.
   Wait until the upload finishes.

The notebook expects exactly: `My Drive/BOTOTOY_2.0/Fruit.v1i.yolov8.zip`.

## 2. Open the notebook in Colab

Open this link (same Google account):

**https://colab.research.google.com/github/imPickleRiick/BOTOTOY-2.0-Fruit-Detection-and-Identification-Using-YOLO/blob/v2-assistive/notebooks/train_fruit31_colab.ipynb**

## 3. Turn on the GPU

Menu **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save**.

## 4. Run the cells in order

Click into the first code cell and press **Shift + Enter** for each cell, waiting for each to finish:

| Cell | What it does | What you should see |
|---|---|---|
| 1. Check the GPU | Confirms the T4 is active | `GPU: Tesla T4` |
| 2. Connect Google Drive | Asks for Drive access: choose your account and **Allow** | `Runs will be saved in: /content/drive/MyDrive/BOTOTOY_2.0/runs` |
| 3. Get the project code | Downloads the `v2-assistive` branch and installs Ultralytics | `Code version: ...` and `Ultralytics 8.4.215` |
| 4. Prepare the dataset | Downloads the original dataset, copies the ZIP from Drive and rebuilds the committed combined split (6–8 min) | `Split: {'train': 13437, 'val': 2254, 'test': 1432}` |
| 5. Train YOLO11n | The actual training (about 2 hours) | A progress line per epoch, up to `50/50` (it may stop earlier when it stops improving) |
| 5b. Resume | **Skip** unless the session disconnected (see below) | `RESUME is False - skipping.` |
| 6. Check and download | Shows the results and downloads the ZIP | Browser downloads `bototoy_fruit31_training_results.zip` |

Keep the Colab tab open while it trains. Colab disconnects tabs that are idle for a long time, so check on it
every 20–30 minutes.

## 5. If something goes wrong

| Problem | Fix |
|---|---|
| `No GPU!` | Step 3 was skipped: Runtime → Change runtime type → T4 GPU, then run the cells again |
| `Upload Fruit.v1i.yolov8.zip to ... first` | The ZIP is not in `My Drive/BOTOTOY_2.0/`, or the upload has not finished |
| `Split differs from the repository!` | The ZIP is a different export; it must be "Fruit" by trucly, version 1, YOLOv8 format |
| `CUDA out of memory` | In cell 5 set `BATCH = 8` and run cell 5 again |
| Session disconnected during training | Reconnect (T4 GPU), run cells 1–4 again, then in cell 5b set `RESUME = True` and run it. Training continues from the last saved epoch on Drive. |
| "You cannot currently connect to a GPU" (free-tier limit) | Wait a few hours, or use another team member's Google account (they need the ZIP in their own Drive) |

## 6. Bring the result back

Put `bototoy_fruit31_training_results.zip` in your Downloads folder and tell Claude, or run this in the project
folder:

```bash
python -m src.yolo.training_report path/to/bototoy_fruit31_training_results.zip --config config/config_fruit31.yaml --out outputs/yolo_fruit31/training
```

This verifies and installs `models/trained/bototoy_fruit31_yolo11n.pt`. The Fruit Reader then switches to the
31-fruit model automatically the next time the app starts.
