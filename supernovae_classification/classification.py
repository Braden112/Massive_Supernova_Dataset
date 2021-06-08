# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:33:41 2021

@author: blgnm
"""


def Classify_SN(Train, y, evaluate = True , smot = True,
                                   Ada = True, KNN = False,roc = True, Rand = False,
                                   grid = False, n = 1, fold = 3,n_components = 20, 
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
    if Train is not None:
        TrainingData, u = pd.concat([pd.DataFrame(data=y),pd.DataFrame(data=Train)],axis=1).reset_index(drop=True), y
    #else *** Remember to make this load in default training data
    svc = RandomForestClassifier(n_estimators = 30, min_samples_split = 6)
    TrainingData = TrainingData.sample(frac = 1).reset_index(drop=True)
    if kwargs:
        if Ada ==True:
            classifier = AdaBoostClassifier(**kwargs)
        if KNN == True:
            classifier = KNeighborsClassifier(**kwargs)
        if Rand == True:
            #classifier = RandomForestClassifier(**kwargs)
            classifier = BalancedRandomForestClassifier(**kwargs)
      
          
    else:
        classifier=AdaBoostClassifier(base_estimator=svc,n_estimators=30, learning_rate =2)
    #classifier = KNeighborsClassifier(n_neighbors=1500)

    if evaluate == True:
        
        if smot == True:
            pipeline = imbpipeline(steps = [['smote', SMOTE()],['classifier', BalancedRandomForestClassifier()]])
        if smot == False:
            from sklearn.ensemble import RandomForestRegressor
            pipeline = imbpipeline(steps = [['classifier', BalancedRandomForestClassifier(**kwargs)]])
        
       
        
        stratified_kfold = StratifiedKFold(n_splits=fold, shuffle=True)
        repeatstratified_kfold = RepeatedStratifiedKFold(n_splits=fold, n_repeats=n)
        cross = cross_validate(pipeline, np.array(TrainingData.iloc[:,1:]),np.array(TrainingData.iloc[:,0]),scoring = metric, cv = repeatstratified_kfold, n_jobs = -1)
        print(f'The mean {metric} over {fold} fold stratified crossvalidation repeated {n} times is {np.mean(cross["test_score"])}, with a standard deviation of {np.std(cross["test_score"])}')
        #c = list()
        #for i in range(10):
         #   y_pred = cross_val_predict(pipeline, np.array(TrainingData.iloc[:,1:]),np.array(TrainingData.iloc[:,0]), cv = stratified_kfold, n_jobs = -1)
          #  y_prob = cross_val_predict(pipeline, np.array(TrainingData.iloc[:,1:]),np.array(TrainingData.iloc[:,0]), cv = stratified_kfold, n_jobs = -1,method = 'predict_proba')
           # prediction  = pd.concat([pd.DataFrame(y_pred,columns = ['class']),pd.DataFrame(y_prob,columns=['Type1a','Type2','Type1bc','SLSN'])],axis=1)
            #conf_mat = confusion_matrix(TrainingData.iloc[:,0], y_pred)
            #c.append(conf_mat)
        #conf_mat = (sum(c))/(10)
        y_pred = cross_val_predict(pipeline, np.array(TrainingData.iloc[:,1:]),np.array(TrainingData.iloc[:,0]), cv = stratified_kfold, n_jobs = -1)
        y_prob = cross_val_predict(pipeline, np.array(TrainingData.iloc[:,1:]),np.array(TrainingData.iloc[:,0]), cv = stratified_kfold, n_jobs = -1,method = 'predict_proba')
        prediction  = pd.concat([pd.DataFrame(y_pred,columns = ['class']),pd.DataFrame(y_prob,columns=['Type1a','Type2','Type1bc','SLSN'])],axis=1)
        
        
        classes = ['Type1a', 'Type2', 'Type1bc', 'SLSN']
        b = prediction.replace([0,1,2,3], classes)
        
        for i in classes:
            print(f'The mean probability for {i} is {np.mean(b[b.iloc[:,0]==i][i])} with a standard deviation of {np.std(b[b.iloc[:,0]==i][i])}')
        
        g = pd.concat([b[b.iloc[:,0]=='Type1a']['Type1a'].reset_index(drop=True),b[b.iloc[:,0]=='Type2']['Type2'].reset_index(drop=True),b[b.iloc[:,0]=='Type1bc']['Type1bc'].reset_index(drop=True),b[b.iloc[:,0]=='SLSN']['SLSN'].reset_index(drop=True)],axis=1)
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
        
        val = TrainingData.iloc[:,0]
        
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
        conf_mat = confusion_matrix(TrainingData.iloc[:,0], y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        disp.plot(cmap = 'Blues')
        
        conf_mat1 = confusion_matrix(TrainingData.iloc[:,0], y_pred, normalize = 'pred')
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat1)
        disp.plot(cmap = 'Blues')
        
        if grid == True:
            from sklearn.model_selection import RandomizedSearchCV
            #clf = GridSearchCV(pipeline, param_grid, n_jobs = -1, cv = stratified_kfold, scoring = 'f1_micro')
            clf = RandomizedSearchCV(pipeline, param_grid, n_iter=1000, n_jobs = -1, cv = stratified_kfold, scoring = metric,verbose = 10)

            clf.fit(TrainingData.iloc[:,1:], TrainingData.iloc[:,0])

        
           
        
        plot_confusion_matrix1(conf_mat, ['Type 1a','Type 2', 'Type 1b/c', 'SLSN'], cmap = 'Blues')
  
    Classifier = pipeline.fit(TrainingData.iloc[:,1:], TrainingData.iloc[:,0])
    
    if grid == False:
        return Classifier 
    if grid == True:
        return clf