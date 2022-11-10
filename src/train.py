from tflite_model_maker.config import ExportFormat
from tflite_model_maker import object_detector

import tensorflow as tf

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
        model = object_detector.create(train_data=self.data[0], 
                                model_spec=spec, 
                                validation_data=self.data[1], 
                                epochs=epochs, 
                                batch_size=batch_size, 
                                train_whole_model=True)
        model.evaluate(self.data[2])
        TFLITE_FILENAME = '{m_name}+_{ep}e+_{btc}b.tflite'.format(m_name = spec.config.backbone_name, ep= epochs, btc = batch_size)
        LABELS_FILENAME = 'face-labels.txt'
        model.export(export_dir=dir, tflite_filename=TFLITE_FILENAME, label_filename=LABELS_FILENAME,
        export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])
        return model

def main():

    label_map = {1: "face"}
    trainer = model_trainer(
        label_map, "dataset/train", "dataset/test", "dataset/validation", "small"
    )
    trainer.load_data(label_map, "dataset/train", "dataset/test", "dataset/validation")
    
    trainer.train_model(epochs =10, batch_size= 32)

    trainer.save_model('models',trainer.train_model(10,32))
    
if __name__ == "__main__":
    main()
    
