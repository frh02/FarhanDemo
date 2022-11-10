import sys
#from ossaudiodev import control_names
import numpy as np
import os
#import argparse
import cv2
import os
import datetime 
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
import time
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import object_detector

import tensorflow as tf
#from torch import R

#temp until proper logger integrated.
log = print('G')

class model_trainer():
    def __init__(self, labelmap, traindir, testdir, valdir, modelsize):
        '''
        modelsize = small, medium, large (strings)
        '''
        self.labelmap = labelmap
        self.traindir = traindir
        self.testdir = testdir 
        self.valdir = valdir
        self.modelsize = modelsize
        self.dataloaded = False
        # train, test, validate 
        self.data = [[],[],[]]
        #self.model = 
        
    def load_data(self, label_map,traindir, testdir, valdir):
        """
        - Need: [train, test, validation] x [images, annotations]
        - Each dir should have images folder & annotations folder (pascal format)
        """    
        train_data = object_detector.DataLoader.from_pascal_voc(
            traindir+'/images', traindir+'/annotations', label_map=label_map
        )
        validation_data = object_detector.DataLoader.from_pascal_voc(
            testdir+'/images', testdir+'/annotations', label_map=label_map
        )
        test_data = object_detector.DataLoader.from_pascal_voc(
            valdir+'/images', valdir+'/annotations', label_map=label_map
        )
        self.dataloaded = True
        # except Exception as e:
        #     log("Unable to load data", e)
        #     sys.Exit(0)
        self.data[0] = train_data
        self.data[1] = test_data
        self.data[2] = validation_data
        print(f'train count: {len(train_data)}')
        print(f'validation count: {len(validation_data)}')
        print(f'test count: {len(test_data)}')
        

    def train_model(self,epochs, batch_size):
        '''
        Create a model for training.
        '''
        spec = object_detector.EfficientDetLite0Spec()
        spec.config.max_instances_per_image = 6000 
        print(spec.config)
        model = object_detector.create(train_data=self.data[0], 
                                model_spec=spec, 
                                validation_data=self.data[1], 
                                epochs=epochs, 
                                batch_size=batch_size, 
                                train_whole_model=True)
        model.evaluate(self.data[2])
        print(model.type)
        return model

    def save_model(dir, model):
        '''
        Save the model.
        '''
        TFLITE_FILENAME = 'efficientdet-lite0_small_ax_face_300e_64b.tflite'
        LABELS_FILENAME = 'face-labels.txt'
        model.export(export_dir=dir, tflite_filename=TFLITE_FILENAME, label_filename=LABELS_FILENAME,
        export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])

class inference_engine():
    def __init__(self, modeldir, input_path, output_path, size):
        self.modeldir = modeldir
        self.size = size 
        self.input_path = input_path
        self.output_path = output_path
    def append_objs_to_img(cv2_im, inference_size, objs, labels):
        height, width, channels = cv2_im.shape
        scale_x, scale_y = width / inference_size[0], height / inference_size[1]
        for obj in objs:
            bbox = obj.bbox.scale(scale_x, scale_y)
            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)

            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        return cv2_im
    def interpreter(self, default_model,labels,threshold, top_k =3):
        model = os.path.join(self.modeldir, default_model)
        interpreter = make_interpreter(model)
        interpreter.allocate_tensors()
        labels = read_label_file(labels)
        inference_size = input_size(interpreter)
    
        #cap = cv2.VideoCapture(args.camera_idx)
        cap = cv2.VideoCapture(self.input_path +'/reiner.avi')
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        recording = False
        fcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (frame_width, frame_height)
        result = cv2.VideoWriter(self.output_path+'/MAR3_eff0_300e_b64_ax_reiner_face.avi',fcc, 30, size)

        while cap.isOpened():
            ret, frame = cap.read()
            start_time = datetime.datetime.now()
            num_frames = 0
            start = time.time()
            if not ret:
                break
            cv2_im = frame

            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
            run_inference(interpreter, cv2_im_rgb.tobytes())
            objs = get_objects(interpreter, threshold)[:top_k]
        
            cv2_im = inference_engine.append_objs_to_img(cv2_im, inference_size, objs, labels)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            fps = num_frames / elapsed_time
            cv2.putText(cv2_im, "FPS: " + str("{0:.2f}".format(fps)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
            print("fps:",1/(time.time()-start))
            cv2.imshow('frame', cv2_im)
            k = cv2.waitKey(1) & 0xff
            if k == ord('r') and recording is False:
                print('recording start')
                recording = True 
            if recording:
                result.write(cv2_im)
            if k == ord('e'):
                print('recording end')
                recording = False 
                result.release()
            
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

        cap.release()
        cv2.destroyAllWindows()
        
    
    
    
def main():
    label_map = {1: "face"}
    trainer = model_trainer(
        label_map, "dataset/train", "dataset/test", "dataset/validation", "small"
    )
    trainer.load_data(label_map, "dataset/train", "dataset/test", "dataset/validation")
    trainer.train_model(epochs =10, batch_size= 32)
    #trainer.save_model('models',trainer.train_model(10,32))
    interpret = inference_engine("models","input_videos", "output_videos","small")
    labels = 'face_labels.txt'

    interpret.interpreter('efficientdet-lite0_small_ax_face_300e_64b_edgetpu.tflite',labels, 0.1, 3 )

if __name__ == "__main__":
    main()
    