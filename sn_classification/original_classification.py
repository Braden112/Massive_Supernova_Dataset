# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:33:41 2021

@author: blgnm
"""
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def Classify_SN(Train, y, evaluate = True , smot = False,
                                   Ada = False, KNN = False,roc = True, Rand = True,
                                   grid = False, n = 10, fold = 3,n_components = 20, 
                                   metric = 'accuracy', param_grid = None,**kwargs):
    """
    

    Parameters
    ----------
    Trains for Supernova Type Classification
    
    Train : Pandas Data Frame, optional
        Provide your own wavelet coefficients. The default is None.
    y : Pandas Data Frame, optional
        Provide data labels. The default is None.
    evaluate : Boolean, optional
        Choose whether or not to show model performance. The default is True.
    Ada : Boolean, optional
        Choose to use Ada Boosted Random Forrest. The default is True.
    KNN : Boolean, optional
        Choose to use K-nearest neighbors. The default is False.
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    Exception
        If you set both KNN and Ada to false, raises an error.

    Returns
    -------
    Function
        Trained Classifier.

    """
    Train = pd.concat([pd.DataFrame(y).reset_index(drop=True), Train.reset_index(drop=True)],axis=1)
    Train = Train.sample(frac=1).reset_index(drop=True)
    
    #classifier = BalancedRandomForestClassifier(**kwargs)
    classifier = BalancedRandomForestClassifier(criterion = 'entropy', 
                                                      max_features = 'sqrt', 
                                                      n_estimators = 1000, n_jobs = -1,
                                                      max_depth = 15, min_samples_leaf = 1, min_samples_split = 2,replacement = True, class_weight = 'balanced_subsample')

    stratified_kfold = StratifiedKFold(n_splits=fold, shuffle=True)
    repeatstratified_kfold = RepeatedStratifiedKFold(n_splits=fold, n_repeats=n)
    cross = cross_validate(classifier, np.array(Train.iloc[:,1:]),np.array(Train.iloc[:,0]).ravel(),scoring = metric, cv = repeatstratified_kfold, n_jobs = -1)
    print(f'The mean {metric} over {fold} fold stratified crossvalidation repeated {n} times is {np.mean(cross["test_score"])}, with a standard deviation of {np.std(cross["test_score"])}')
    
    y_pred = cross_val_predict(classifier, np.array(Train.iloc[:,1:]),np.array(Train.iloc[:,0]).ravel(), cv = stratified_kfold, n_jobs = -1)
    y_prob = cross_val_predict(classifier, np.array(Train.iloc[:,1:]),np.array(Train.iloc[:,0]).ravel(), cv = stratified_kfold, n_jobs = -1,method = 'predict_proba')
    prediction  = pd.concat([pd.DataFrame(y_pred,columns = ['class']),pd.DataFrame(y_prob,columns=['Type1a','Type2','Type1bc','SLSN'])],axis=1)
    

    classes = [0,1,2,3]
    b = prediction
    #for i in classes:
     #   print(f'The mean probability for {i} is {np.mean(b[b.iloc[:,0]==i][i])} with a standard deviation of {np.std(b[b.iloc[:,0]==i][i])}')
    g = pd.concat([b[b.iloc[:,0]==0]['Type1a'].reset_index(drop=True),b[b.iloc[:,0]==1]['Type2'].reset_index(drop=True),b[b.iloc[:,0]==2]['Type1bc'].reset_index(drop=True),b[b.iloc[:,0]==3]['SLSN'].reset_index(drop=True)],axis=1)
    g.columns = ['Type Ia', 'Type II', 'Type Ib/c', 'SLSN']
    
    plt.figure(dpi=1200)
    plt.ylim([0,1])
    g.boxplot()
    plt.title('Boxplot of Photometrically Classified Supernova Probabilities')    
    plt.show()
    plt.close()   
    
    plt.figure(dpi=1200)
    plt.hist(prediction[prediction['class']==0]['Type1a'],bins=10,histtype='step',color='red')
    plt.hist(prediction[prediction['class']==1]['Type2'],bins=10,histtype='step',color='blue')
    plt.hist(prediction[prediction['class']==2]['Type1bc'],bins=10,histtype='step',color='green')
    plt.hist(prediction[prediction['class']==3]['SLSN'],bins=10,histtype='step',color='purple')
    plt.title('Probability Distribution of Photometrically Classified Supernovae')
    plt.legend(['Type Ia', 'Type II', 'Type Ib/c', 'SLSN'])
    plt.show()
    plt.close()
    
    val = Train.iloc[:,0]
    
    prediction1 = pd.concat([val,prediction],axis=1)
    ta = prediction1[prediction1.iloc[:,0]==0]
    t2 = prediction1[prediction1.iloc[:,0]==1]
    tbc = prediction1[prediction1.iloc[:,0]==2]
    SLSN = prediction1[prediction1.iloc[:,0]==3]
    
    ta = ta[ta['Type1a'] > .40]
    t2=t2[t2['Type2'] > .40]
    tbc=tbc[tbc['Type1bc'] > .40]
    SLSN=SLSN[SLSN['SLSN'] > .40]
    prediction1 = pd.concat([ta, t2, tbc, SLSN])
    
    conf_mat = confusion_matrix(prediction1.iloc[:,0], prediction1['class'])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot(cmap = 'Blues')
    plot_confusion_matrix1(conf_mat, ['Type 1a','Type 2', 'Type 1b/c', 'SLSN'], cmap = 'Blues')
    
    conf_mat1 = confusion_matrix(prediction1.iloc[:,0], prediction1['class'], normalize = 'pred')
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat1)
    disp.plot(cmap = 'Blues')
    
    conf_mat = confusion_matrix(y, y_pred)
    conf_mat = confusion_matrix(Train.iloc[:,0], y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot(cmap = 'Blues')
    
# =============================================================================
#     conf_mat1 = confusion_matrix(Train.iloc[:,0], y_pred, normalize = 'pred')
#     disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat1)
#     disp.plot(cmap = 'Blues')
# =============================================================================
    
    
   
        
    if grid == True:
        from sklearn.model_selection import RandomizedSearchCV
        #clf = GridSearchCV(pipeline, param_grid, n_jobs = -1, cv = stratified_kfold, scoring = 'f1_micro')
        clf = RandomizedSearchCV(classifier, param_grid, n_iter=1000, n_jobs = -1, cv = stratified_kfold, scoring = metric,verbose = 10)

        clf.fit(Train.iloc[:,1:], Train.iloc[:,0])

        
           
        
    plot_confusion_matrix1(conf_mat, ['Type 1a','Type 2', 'Type 1b/c', 'SLSN'], cmap = 'Blues')
  
    Classifier = classifier.fit(Train.iloc[:,1:],Train.iloc[:,0])
    
    if grid == False:
        return Classifier 
    if grid == True:
        return clf

def plot_confusion_matrix1(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
