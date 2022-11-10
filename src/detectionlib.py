import cv2
import os
import time
import datetime

from imutils.video import FPS
from PIL import Image
from PIL import ImageDraw


from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

class inference_engine():
    def __init__(self, modeldir, input_path, output_path,size, top_k, threshold, labels):
        self.modeldir = modeldir
        self.size = size 
        self.input_path = input_path
        self.output_path = output_path
        self.top_k = top_k
        self.threshold = threshold
        self.labels = labels

    def load_model(self, default_model, labels):
        model = os.path.join(self.modeldir, default_model)
        interpreter = make_interpreter(model)
        interpreter.allocate_tensors()
        labels = read_label_file(labels)

        # self.interpreter = make_interpreter(model)
        #self.inference_size = input_size(self.interpreter)
    
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
    def draw_objects(draw, objs, labels):
        """Draws the bounding box and label for each object."""
        for obj in objs:
            bbox = obj.bbox
            draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                        outline='red')
            draw.text((bbox.xmin + 10, bbox.ymin + 10),
                    '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                    fill='red')

    def run_image_detect(self, model, output_image_dir, input_image_path, count=10):
        labels = read_label_file(self.labels) if self.labels else {}
        interpreter = make_interpreter(model)
        interpreter.allocate_tensors()
        image = Image.open(input_image_path)
        _, scale = common.set_resized_input(
            interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
        for _ in range(count):
            start = time.perf_counter()
            interpreter.invoke()
            inference_time = time.perf_counter() - start
            objs = detect.get_objects(interpreter, self.threshold, scale)
            print('%.2f ms' % (inference_time * 1000))
        if not objs:
            print('No objects detected')    
        for obj in objs:
            print(labels.get(obj.id, obj.id))
            print('  id:    ', obj.id)
            print('  score: ', obj.score)
            print('  bbox:  ', obj.bbox)

        if output_image_dir:
            image = image.convert('RGB')
            inference_engine.draw_objects(ImageDraw.Draw(image), objs, labels)
            image.save(output_image_dir)
            image.show()
        
    def interpreter_video_cam(self, model):
        labels = read_label_file(self.labels) if self.labels else {}
        interpreter = make_interpreter(model)
        interpreter.allocate_tensors()
        inference_size = input_size(interpreter)
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        time.sleep(1)

        fps = FPS().start()

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
            objs = get_objects(interpreter, self.threshold)[:self.top_k]
        
            cv2_im = inference_engine.append_objs_to_img(cv2_im, inference_size, objs, labels)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            fps = num_frames / elapsed_time
            cv2.putText(cv2_im, "FPS: " + str("{0:.2f}".format(fps)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
            print("fps:",1/(time.time()-start))
            cv2.imshow('frame', cv2_im)
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
               break

        cap.release()
        cv2.destroyAllWindows()

    def interpreter_vid(self, default_model):
        model = os.path.join(self.modeldir, default_model)
        interpreter = make_interpreter(model)
        interpreter.allocate_tensors()
        labels = read_label_file(self.labels)
        inference_size = input_size(interpreter)
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
            objs = get_objects(interpreter, self.threshold)[:self.top_k]
        
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
        cap.release()
        cv2.destroyAllWindows()
        
    
    
    
def main():

    label_map = {1: "face"}
    
if __name__ == "__main__":
    main()
    
