# this is a densenet implementation based on the paper https://arxiv.org/abs/1608.06993 using keras and tensorflow.
#inspiration regarding the implementation was also taken from https://www.youtube.com/watch?v=QKtoh9FJIWQ&t=1747s and https://github.com/cmasch/densenet



#arguments: 
#in_shape= shape of the input
#nb_classes= number of classes in the output
#nb_denseblocks= number of dense blocks
#nb_layers= number of layers in a block
#growth_rate= growth rate of the architecture
#depth= depth of the neural network
#compression= compression ratio (0-1)
#bottleneck= (bool) Wether to include bottleneck in the system or not.
#dropout= (0-1) the dropout to be given
#weight_decay= weight decay to be applied to the model

#returns created model with all the parameters.

import tensorflow as tf
import tensorflow.keras.layers as layers



#returns a keras model.
class Densenet:
    def __init__(self,in_shape=[32,32,3], nb_classes=10,nb_denseblocks=3, nb_layers=None, growth_rate=12,depth=40,
                compression=.5, bottleneck=True,weight_decay=10e-4,dropout=0):
        
        self.in_shape= in_shape
        self.nb_classes= nb_classes
        self.nb_denseblocks= nb_denseblocks
        self.nb_layers=[] # list of number of layers per dense block.
        if nb_layers==None:
            nb_layers= (depth-4)//nb_denseblocks
            
            if bottleneck==True:
                nb_layers=nb_layers//2
            for _ in range(nb_denseblocks):
                self.nb_layers.append(nb_layers)
        else:
            for i in range(nb_denseblocks):
                self.nb_layers.append(nb_layers[i]) 
                
        self.growth_rate= growth_rate #number of filters in each layers
        self.detph= depth #depth of the Dense net
        self.compression=compression
        self.bottleneck= bottleneck
        self.weight_decay=weight_decay
        self.dropout=dropout
       
    def create_model(self):
        
        #The basic block of densenet compromising of batchnorm, relu and convolution. If bottleneck is true the 1*1 conv is applied before.
        def bn_conv_relu(x,k_size):
            x=layers.BatchNormalization()(x)
            x=layers.ReLU()(x)
            if k_size[0]==1:
                bn_width=4
                x= layers.Conv2D(self.growth_rate*bn_width,kernel_size=k_size,strides=(1,1),padding='same',
                                kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
                if self.dropout:
                    x=layers.Dropout(self.dropout)(x)
            else:
                x= layers.Conv2D(self.growth_rate,kernel_size=k_size,strides=(1,1),padding='same',
                                kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
                if self.dropout:
                    x=layers.Dropout(self.dropout)(x)
            
            return x
       
    
        #Denseblock method to create denseblocks by repeating convolution layers and concatinating them 
        def DenseBlock(x,nb_layers):
            for _ in range(nb_layers):
                y=x
                if self.bottleneck:
                    y=bn_conv_relu(x,(1,1))
                y=bn_conv_relu(y,(3,3))
                x=layers.Concatenate(axis=-1)([x,y])
            return x
        
        #the transition layer. If compression factor is given then the number of channels is multiplied by that factor.
        def Transition_layer(x, compression):
            x=layers.BatchNormalization()(x)
            x=layers.Conv2D(int(x.shape[-1]*compression),kernel_size=(1,1),strides=(1,1),padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
            if self.dropout:
                x=layers.Dropout(self.dropout)(x)
                
            x=layers.AveragePooling2D((2,2),padding ='valid')(x)
            return x
        
        
        
        
        Input= layers.Input(self.in_shape) # the input layer
        if self.bottleneck:
            x=layers.Conv2D(2*self.growth_rate,kernel_size=(1,1),strides=(1,1),padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(Input)# first 3*3 convolution.
            if self.dropout:
                x=layers.Dropout(self.dropout)(x)
        else:
            x=layers.Conv2D(16,kernel_size=(1,1),strides=(1,1),padding='same',
                            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(Input)
            if self.dropout:
                x=layers.Dropout(self.dropout)(x)
        
        for dense_block in range(self.nb_denseblocks):
            nb_layers=self.nb_layers[dense_block]
            x1=DenseBlock(x,nb_layers)
            x=Transition_layer(x1,self.compression)
        x=layers.BatchNormalization()(x1)
        x=layers.ReLU()(x)
        x=layers.GlobalAveragePooling2D()(x)     #global averaging layer
        preds=layers.Dense(self.nb_classes,activation='softmax')(x) #softmax classifier with output classes given by nb_channels
        
        return tf.keras.Model(inputs=Input,outputs=preds)
        