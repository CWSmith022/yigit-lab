# yigit-lab

## AGONS
![image](https://user-images.githubusercontent.com/66841220/190235928-e1e64e4a-7288-4821-be5b-06c6d72fd163.png)

Code used for projects in the Yigit lab at the University at Albany.
Primary folder is for machine learning powered nanosensor array (MILAN) based on our published reports. In this folder, the in-house programmed algorithmically guided optical nanosensor selector (AGONS) can be found and combined with MILAN or any other nanosensor array system. Please reference the below reports if you use our code.

## Running the Code:
Initiating AGONS for training and cross-validation:
```assembly
from AGONS_nano.AGONSModule import AGONS
agons = AGONS(k_max = x_train.shape[1], cv_method='Repeated Stratified K Fold', cv_fold=10, random_state = 10)
agons.activate(x_train, y_train.squeeze(), x_val, y_val.squeeze())
```

AGONS modeling will then initiate. The following sequence can be used for generating figures and exploring the modeling:
```
agons.featuredisplay()
agons.featureselect()
agons.pca_transform()
agons.pca_diagnostic()
agons.pca2D(loadings = False)
agons.pca3D()
```

After modeling, you can select the parameter set that best fits you requirements for modeling. Use 'b' to denote a parameter set.
```
b = dict(agons.parameter_table().iloc[0, 0:9])
b
```

Set the final model parameters and re-input training data testing data:
```
agons.set_final_model(model_params=b, 
                      x_fit = x_train, 
                      y_fit = y_train.squeeze())
                      ```

Predict on future cases on test set:
```
y_pred = agons.predict(x_test, y_test)
```

Predict probability on future cases on test set:
```
proba = agons.predict_probe(x_test, y_test)
```

Reports:

https://pubs.acs.org/doi/abs/10.1021/acs.analchem.1c04379

https://pubs.acs.org/doi/10.1021/acsfoodscitech.2c00181

