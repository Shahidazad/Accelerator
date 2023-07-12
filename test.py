import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import pandas as pd
from utility import subplot,mapFeature, plotDecisionBoundary,costFunc,sigmoid,y_pred,plot_decision_boundary
import utility
from ipywidgets import widgets
from tabulate import tabulate
from sklearn.svm import SVC



def logistic_regression(df: pd.DataFrame):
    """
    Function for Comparion  Classification model
    Parameters: df:(data Frame)
    return: Graphs and Reports in tabular form
    """
    x_1= 'Plasma glucose concentration' #@param {type:"string"}
    x_2='Body mass index (weight in kg/(height in m)^2)' #@param {type:"string"}
    X=df[[x_1,x_2]]
    target='target' #@param {type:"string"}
    y=df[target]

    # Performing feature scaling
    scale=StandardScaler()
    X=scale.fit_transform(X)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

    print("""
    Which model you have to use :
    1.Logistic Regression
    2.For high Degree of logistic Regression
    3.Support Vector Machine
    """)

    degree=int(input("choice the model :\n "))

    if degree == 1:
        # Initialize the global variable `y_pred` to None
        global y_pred
        y_pred = None

        # Iterate over the degrees from 1 to 4
        for degree in range(1, 5):
          

            poly_df=pd.DataFrame(X_test)
            # Get the features
            X_mapf= poly_df.iloc[:, :2]


            if degree==degree:
                X_poly = mapFeature(X_mapf.iloc[:, 0], X_mapf.iloc[:, 1], degree)
                # Get the target variable
                y_poly = y_test

                # Set initial values for our parameters
                initial_theta = np.zeros(X_poly.shape[1]).reshape(X_poly.shape[1], 1)

                res = minimize(costFunc, initial_theta, args=(X_poly, y_poly))
                
                # our optimizated coefficients
                theta = res.x

                



                poly_df_with_y=poly_df
                poly_df_with_y[target]=y_test.values
                # Create boolean masks for positive and negative target values
                pos=poly_df_with_y[target]==1
                neg=poly_df_with_y[target]==0



                # # calling function from utility file
                # fig,axes,text=subplot(poly_df,pos,neg,degree)

                # # Plot the decision boundary
                # plotDecisionBoundary(theta, degree, axes)
                # plt.show()
                # Plot Decision boundary
                fig, axes = plt.subplots(figsize=(12,7));
                # axes.figure(figsize=(12,7))
                axes.set_xlabel('Feature 1')
                axes.set_ylabel('Feature 2')
                axes.scatter(poly_df.loc[pos, 0], poly_df.loc[pos, 1], color = 'r', marker='x', label='diabetic')
                axes.scatter(poly_df.loc[neg, 0], poly_df.loc[neg, 1], color = 'g', marker='o', label='normal')
                axes.legend(title='Legend', loc = 'best' )
                if degree==1:
                    axes.set_title('linear Decision Boundary for logistic regression ')
                    text = '\033[1m\033[34mlinear Decision Boundary Report\033[0m'

                elif degree==2:
                    axes.set_title('Quadratic Decision Boundary for logistic regression ')
                    text='\033[1m\033[34mQuadratic Decision Boundary Report\033[0m'
                elif degree==3:
                    axes.set_title('cubic Decision Boundary for logistic regression ')
                    text='\033[1m\033[34mcubic Decision Boundary Report\033[0m'
                else:
                    axes.set_title(f'{degree}th Degree  Decision Boundary for logistic regression ')
                    text=f'\033[1m\033[34m{degree}th Degree Decision Boundary Report\033[0m'
                plotDecisionBoundary(theta, degree, axes)

                


            # Calculate accuracy score
            
            accuracy = accuracy_score(y_test, utility.y_pred)

            # Generate classification report
            report = classification_report(y_test, utility.y_pred)

            # Create table headers and data
            headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
            rows = []
            for line in report.split("\n")[2:-5]:
                class_info = line.split()
                class_name = class_info[0]
                precision, recall, f1_score, support = class_info[1:]
                rows.append([class_name, precision, recall, f1_score, support])
            rows.append(["Accuracy Score", "", "", "", accuracy])

            # Print table using tabulate
            print(text)
            print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

    elif degree==2:


        degree=int(input("How much degree curve you want :\n "))

        poly_df=pd.DataFrame(X_test)
        # Get the features
        X_mapf= poly_df.iloc[:, :2]

        X_poly = mapFeature(X_mapf.iloc[:, 0], X_mapf.iloc[:, 1], degree)
        # Get the target variable
        y_poly = y_test

        # Set initial values for our parameters
        initial_theta = np.zeros(X_poly.shape[1]).reshape(X_poly.shape[1], 1)

        res = minimize(costFunc, initial_theta, args=(X_poly, y_poly))
        # our optimizated coefficients
        theta = res.x

        poly_df_with_y=poly_df
        poly_df_with_y['target']=y_test.values

        pos=poly_df_with_y['target']==1
        neg=poly_df_with_y['target']==0


        # Plot Decision boundary
        fig, axes = plt.subplots(figsize=(12,7));
        # axes.figure(figsize=(12,7))
        axes.set_xlabel('Feature 1')
        axes.set_ylabel('Feature 2')
        axes.set_title(f'{degree}th Degree  Decision Boundary for logistic regression ')
        axes.scatter(poly_df.loc[pos, 0], poly_df.loc[pos, 1], color = 'r', marker='x', label='diabetic')
        axes.scatter(poly_df.loc[neg, 0], poly_df.loc[neg, 1], color = 'g', marker='o', label='normal')
        axes.legend(title='Legend', loc = 'best' )
        plotDecisionBoundary(theta, degree, axes)

        # Calculate accuracy score

        accuracy = accuracy_score(y_test, utility.y_pred)

        # Generate classification report
        report = classification_report(y_test, utility.y_pred)

        # Create table headers and data
        headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        rows = []
        for line in report.split("\n")[2:-5]:
            class_info = line.split()
            class_name = class_info[0]
            precision, recall, f1_score, support = class_info[1:]
            rows.append([class_name, precision, recall, f1_score, support])
        rows.append(["Accuracy Score", "", "", "", accuracy])

        # Print table using tabulate
        print('\033[1m\033[34mMulti Decision Boundary Report\033[0m' )
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

    else:
        model = SVC(kernel = 'linear')
        model.fit(X_train, y_train)
        y_pred=model.predict(X_test)


        theta = np.concatenate((model.intercept_, model.coef_.flatten()))
        df_test=pd.DataFrame(X_test)
        df_test['target']=np.where(y_test==0,0,1)
        pos=df_test.target==1
        neg=df_test.target==0


        # Plot Decision boundary
        fig, axes = plt.subplots(figsize=(12,7));
        # axes.figure(figsize=(12,7))
        axes.set_xlabel('Feature 1')
        axes.set_ylabel('Feature 2')
        axes.set_title('Linear Decision Boundary for SVM  ')
        text = '\033[1mSVM Decision Boundary Report\033[0m'

        axes.scatter(df_test.loc[pos, 0], df_test.loc[pos, 1], color = 'r', marker='x', label='diabetic')
        axes.scatter(df_test.loc[neg, 0], df_test.loc[neg, 1], color = 'g', marker='o', label='normal')
        axes.legend(title='Legend', loc = 'best' )
        # plotDecisionBoundary(theta,1,axes)


        accuracy = accuracy_score(y_test, y_pred)

        # Generate classification report
        report = classification_report(y_test, y_pred)


        # Create table headers and data
        headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        rows = []
        for line in report.split("\n")[2:-5]:
            class_info = line.split()
            class_name = class_info[0]
            precision, recall, f1_score, support = class_info[1:]
            rows.append([class_name, precision, recall, f1_score, support])
        rows.append(["Accuracy Score", "", "", "", accuracy])

        # Print table using tabulate
        print(text)
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

        plotDecisionBoundary(theta,1,axes)

        # for poly SVC
        model = SVC(kernel='poly', degree=3,C=0.4)
        model.fit(X_train, y_train)
        y_pred=model.predict(X_test)



        plot_decision_boundary(model, X_test, y_test,df_test,pos,neg)

        accuracy = accuracy_score(y_test, y_pred)

        # Generate classification report
        report = classification_report(y_test, y_pred)


        # Create table headers and data
        headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        rows = []
        for line in report.split("\n")[2:-5]:
            class_info = line.split()
            class_name = class_info[0]
            precision, recall, f1_score, support = class_info[1:]
            rows.append([class_name, precision, recall, f1_score, support])
        rows.append(["Accuracy Score", "", "", "", accuracy])

        # Print table using tabulate
        print(text)
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

    








