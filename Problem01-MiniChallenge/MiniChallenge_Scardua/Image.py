'''
This file creates a new class from keras ImageDataGenerator class that can
generate batchs from pandas.
Author: Arthur Scardua
Date: set 2018
'''

import pandas as pd
import numpy as np

from keras.preprocessing.image import ImageDataGenerator, Iterator #Image generator class

class ImageDataGeneratorPlus(ImageDataGenerator):
    '''
    A new method for ImageDataGenerator that creates a generator from pandas.
    '''
    def __init__(self, *args,**kargs):
        super(ImageDataGeneratorPlus,self).__init__(*args,**kargs)
    
    def flow_from_pandas(self,pandas,directory,*args,**kargs):
        '''
        flow_from_pandas(self,pandas,directory,target_size=(150, 150), batch_size=16,class_mode='categorical')
        
        Method for creating a Batch from a panda DataFrame with files names and its labels. 
        
        '''
        if not directory:
            directory = '.'
        elif directory[-1]=='/':
            directory = directory[:-1]
        generator = self.flow_from_directory('./',*args,**kargs)
        keys = pandas.keys()
        classes = pandas[keys[1]].unique()
        classes.sort()
        generator.num_classes = len(classes)
        generator.class_indices = dict(zip(classes, range(len(classes))))
        generator.filenames = list(directory+'/'+pandas[keys[0]].values)
        generator.classes = np.array([generator.class_indices[x] for x in pandas[keys[1]] ],
                                   dtype='int32')
        generator.n = generator.samples = len(generator.filenames)
        generator.index_generator = generator._flow_index()
        print('Actually, found %d images belonging to %d classes' % 
             (generator.samples, generator.num_classes))
        return generator
