import sys
sys.path.insert(1,'src')
from detectionlib import inference_engine

def main():
    labels = 'face_labels.txt'
    interpret_vid = inference_engine("models","results/input_videos", "results/output_videos","small", 10, 0.1, labels)
    interpret_vid.interpreter_video_cam('models/efficientdet-lite0_small_ax_face_300e_64b_edgetpu.tflite')

if __name__ == '__main__':
  main()
