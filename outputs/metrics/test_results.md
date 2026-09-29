# Qualitative test results

Model: `best.pt`, confidence threshold: 0.25
Observations are to be written by you after looking at the output images.

## Clear image, common objects (`test1_clear*`)
- **Input:** `test1_clear.png` -> output `test1_clear_pred.png`
  - pedestrian: 25.3%
- **Observations:** _TODO: fill in after viewing the image_

## Multiple objects (`test2_multiple*`)
- **Input:** `test2_multiple.png` -> output `test2_multiple_pred.png`
  - car: 89.0%
  - car: 81.4%
  - car: 73.9%
  - car: 35.8%
  - car: 25.9%
- **Observations:** _TODO: fill in after viewing the image_

## Small / distant objects (`test3_small*`)
- **Input:** `test3_small.png` -> output `test3_small_pred.png`
  - pedestrian: 25.3%
- **Observations:** _TODO: fill in after viewing the image_

## Object outside trained classes (`test4_unseen*`)
- **Input:** `test4_unseen.jpg` -> output `test4_unseen_pred.png`
  - (no detections)
- **Observations:** _TODO: fill in after viewing the image_

## Low quality / partially obscured (`test5_lowquality*`)
- **Input:** `test5_lowquality_auto.png` -> output `test5_lowquality_auto_pred.png`
  - (no detections)
- **Observations:** _TODO: fill in after viewing the image_
