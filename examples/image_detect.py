import sys
sys.path.insert(1,'src')
from detectionlib import inference_engine

def main():
  labels = 'face_labels.txt'
  interpret_im= inference_engine("models","results/images/2_Demonstration_Demonstration_Or_Protest_2_28_jpg.rf.d0ea6fc65e0327a7c8feea19cf64d608.jpg", "results/output_videos", "small", 10, 0.1, labels)
  interpret_im.run_image_detect('models/efficientdet-lite0_small_ax_face_300e_64b_edgetpu.tflite', 'results/output_image/image_processed.jpg', 'results/images/8_Election_Campain_Election_Campaign_8_399_jpg.rf.75a4075e6f05d90fe9ac52f2841fe80f.jpg', count=10 )

if __name__ == '__main__':
  main()
