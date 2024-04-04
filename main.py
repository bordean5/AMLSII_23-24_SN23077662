import keras
from cassava_classification import *
from tensorflow.keras.models import load_model



if __name__ == "__main__":

    """
    Main function to run the whole project
    in default it will only load the pre-trained model from files as training takes a lot of time (20hours)
    you could train the model by change the first variable from False to True
    """
     
    train=False

    #get data and split into train and test
    train_labels, test_labels=load_split()

    BATCH_SIZE = 16
    STEPS_PER_EPOCH = len(train_labels)*0.8 / BATCH_SIZE
    VALIDATION_STEPS = len(train_labels)*0.2 / BATCH_SIZE
    EPOCHS = 32
    TARGET_SIZE = 512

    # get test generator
    test_generator=test_generate(test_labels,TARGET_SIZE,BATCH_SIZE)



    #load the train and valid generators,create model and train
    if (train):
        base_train_generator,base_validation_generator=image_generate(False,train_labels,TARGET_SIZE,BATCH_SIZE)
        augmented_train_generator,augmented_validation_generator=image_generate(True,train_labels,TARGET_SIZE,BATCH_SIZE)

        baseCNN= create_CNNmodel(TARGET_SIZE)
        improvedCNN=create_CNNmodel(TARGET_SIZE)
        EfficientNet=create_EfficientNetmodel(TARGET_SIZE)

        model_training(baseCNN,"baseCNN", base_train_generator,base_validation_generator,
                          EPOCHS, STEPS_PER_EPOCH,VALIDATION_STEPS)

        class_weight=compute_classweight(train_labels)

        weighted_training(improvedCNN,"improvedCNN", augmented_train_generator,augmented_validation_generator,
                          EPOCHS, STEPS_PER_EPOCH,VALIDATION_STEPS,class_weight)
        weighted_training(EfficientNet,"EfficientNet", augmented_train_generator,augmented_validation_generator,
                          20, STEPS_PER_EPOCH,VALIDATION_STEPS,class_weight)

    #load the pre-trained model
    else:
        baseCNN= load_model('Models/baseCNN.h5',compile=False)
        improvedCNN=load_model('Models/improvedCNN.h5',compile=False)
        EfficientNet=load_model('Models/EfficientNetB3.h5',compile=False)

    baseCNN._name='baseCNN'
    improvedCNN._name='improvedCNN'
    EfficientNet._name='EfficientNet'
     
    
    #build the ensemble model
    inputs = keras.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))
    y1 = baseCNN(inputs)
    y2 = improvedCNN(inputs)
    y3 = EfficientNet(inputs)
    outputs = layers.average([y1, y3])
    ensemble_model = keras.Model(inputs=inputs, outputs=outputs)

    #show the valuation result
    print("Evaluation of base CNN:")
    test(baseCNN, test_generator)
    print("Evaluation of CNN with class_weight and image augmentation:")
    test(improvedCNN, test_generator)
    print("Evaluation of EfficientNet transfer learning model:")
    test(EfficientNet, test_generator)
    print("Evaluation of ensemble model of baseCNN and EfficientNet:")
    test(ensemble_model,test_generator)

    
