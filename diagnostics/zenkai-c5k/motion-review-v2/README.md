# Motion review panels

Left to right: Reference | Bilinear | Historical V3 | C5k. Each column is the same unscaled384px ROI used in the still panels. Each clip contains48 distinct frames at the nominal source rate24000/1001, roughly2 seconds; no looping or interpolation. FFV1/BGR0 keeps the stacked RGB pixels lossless, verified by decoding every stacked RGB frame.

These files prepare manual motion review; no human/blind perceptual verdict or player performance result has been collected. The600s scene transition is retained. A separate full-PNG CPU pass matched the selected full-frame metrics; review encoding itself did not run inference or supply the reported metrics.
