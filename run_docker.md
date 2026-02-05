docker run --gpus all -it \
  --shm-size=8g \
  -v "$(pwd -W)/crack_segmentation:/workspace/crack_segmentation" \
  -v "$(pwd -W)/data:/workspace/data" \
  crack-irn:py310
