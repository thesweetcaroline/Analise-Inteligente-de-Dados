import pandas as pd
import numpy as np
from scipy.sparse import data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
'''
Label Encoder converts nominal values into numbers
    Useful for encoding the levels of categorical features into numeric values.
    Useful when the number of categories is quite large as one-hot encoding can lead to high memory consumption.

One Hot Encoder removes distances between numeric values
    Useful when the categorical feature is not ordinal. 
    It allows the representation of categorical data to be more expressive, 
        since there will originate a collumn for each differnt feature value

Standard Scaler makes the standardization between numeric values. 
    Useful when we have different scales and want to distribute values in a standard distribution
'''
class myClass:
    descriptive_train = [[]]
    descriptive_test = [[]]
    target_train = [[]]
    target_test = [[]]

    def __init__(self):
        # Get data from breast-cancer dataset 
        # Initialize breast cancer descriptive and target
        self.df_breast_cancer = pd.read_csv('breast-cancer/dataR2.csv')

        self.bc_descriptive = self.df_breast_cancer.iloc[:, :9].values
        self.bc_target = self.df_breast_cancer.iloc[:, 9].values
        encoder = LabelEncoder()
        n_cols = len(self.bc_descriptive.T)
        for i in range(n_cols):
            self.bc_descriptive[:,i] = encoder.fit_transform(self.bc_descriptive[:,i])
        self.bc_target = encoder.fit_transform(self.bc_target)
        
        # Get data from house-votes-84 dataset 
        # Initialize house votes descriptive and target
        self.df_house_votes = pd.read_csv('voting-records/house-votes-84.csv', 
                                            names=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

        self.hv_descriptive  = self.df_house_votes.iloc[:, 1:16].values
        self.hv_target = self.df_house_votes.iloc[:, 0].values

        encoder = LabelEncoder()
        n_cols = len(self.hv_descriptive.T)
        for i in range(n_cols):
            self.hv_descriptive[:,i] = encoder.fit_transform(self.hv_descriptive[:,i])
        self.hv_target = encoder.fit_transform(self.hv_target)


    def prepare_data_to_model(self, dataset, percentage, use_standard_scaler, use_one_hot_encoder):
        '''
        Dataset:
            0 => Breast Cancer
            1 => House Votes
        '''
        if dataset == 0:
            descriptive = self.bc_descriptive
            target = self.bc_target
            if use_one_hot_encoder:
                column_transformer = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0,1,2,3,4,5,6,7,8])], remainder = 'passthrough')
                descriptive = column_transformer.fit_transform(descriptive).toarray()
        else:
            descriptive = self.hv_descriptive
            target = self.hv_target
            if use_one_hot_encoder:
                column_transformer = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])], remainder = 'passthrough')
                descriptive = column_transformer.fit_transform(descriptive)

        self.descriptive_train, self.descriptive_test, self.target_train, self.target_test = train_test_split(descriptive, target, test_size = percentage/100, random_state = 0)

        if use_standard_scaler:
            standard_scaler = StandardScaler()
            self.descriptive_train[:,:] = standard_scaler.fit_transform(self.descriptive_train[:,:]) #calcula e aplica
            self.descriptive_test[:,:] = standard_scaler.transform(self.descriptive_test[:,:]) #aplica
        
    
    def print_statistics(self, test, prediction):
        accuracy = accuracy_score(test, prediction)
        print("Accuracy: %.4f" % (accuracy))
        matrix = confusion_matrix(test, prediction)
        ##print("\nMatrix:\n" + str(matrix)) 
        ### Plot confusion matrix
        import matplotlib.pyplot as plt
        import plotMAPS as pmaps
        plt.figure(figsize=(3.5,3.5))
        m_labels= {'+', ' -'}
        pmaps.plot_confusion_matrix(matrix, classes = m_labels, title='confusion matrix')
        plt.show()


    def predict_naive_bayes(self):
        classifier = GaussianNB() #Our Model!
        classifier.fit(self.descriptive_train, self.target_train)
        prediction = classifier.predict(self.descriptive_test)
        print(self.print_statistics(self.target_test, prediction))


    def predict_decision_tree(self):
        classifier = DecisionTreeClassifier(criterion="entropy", random_state = 0)
        classifier.fit(self.descriptive_train, self.target_train)
        prediction = classifier.predict(self.descriptive_test)
        print(self.print_statistics(self.target_test, prediction))


    def predict_random_forest(self):
        classifier = RandomForestClassifier(n_estimators=20, criterion="entropy", random_state=0)
        classifier.fit(self.descriptive_train, self.target_train)
        prediction = classifier.predict(self.descriptive_test)
        print(self.print_statistics(self.target_test, prediction))


    def predict_knn(self):
        classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p = 2) #Euclidean Distance
        classifier.fit(self.descriptive_train, self.target_train)
        prediction = classifier.predict(self.descriptive_test)
        print(self.print_statistics(self.target_test, prediction))


def menu():
    while True: 
        print ("\n\n" + 30 * "-" , "Choose one dataset" , 30 * "-")
        print ("1. Breast Cancer")
        print ("2. House Votes 84")
        print ("3. Exit")
        print (67 * "-")
        choice = input("Enter your choice [1-3]: ")
        
        if choice == '1':     
            menu2(0)
        elif choice == '2':
            menu2(1)
        elif choice == '3':
            return
        else:
            input("Wrong option selection. Enter any key to try again..")

def menu2(dataset):
    while True: 
        print ("\n\n" + 30 * "-" , "Enter the percentage of test set" , 30 * "-")
        percentage = input("Percentage: ")
        if percentage == '':
            input("Invalid percentage. Enter a number between 0-100 to try again..")
        elif not percentage.isnumeric:
            input("Invalid percentage. Enter a number between 0-100 to try again..")
        elif float(percentage) > 100 or float(percentage) < 0:
            input("Invalid percentage. Enter a number between 0-100 to try again..")
        else:
            menu3(dataset, float(percentage))
        return
        

def menu3(dataset, percentage):
    print ("\n" + 30 * "-" , "Choose one combination" , 30 * "-")
    print ("1. Label Encoder")
    print ("2. Label Encoder + Standard Scaler")
    print ("3. Label Encoder + One Hot Encoder")
    print ("4. Label Encoder + Standard Scaler + One Hot Encoder")
    print ("5. Exit")
    print (67 * "-")

    loop=True      
    while loop:
            choice = input("Enter your choice [1-5]: ")
            
            if choice == '1': 
                obj.prepare_data_to_model(dataset, percentage, use_standard_scaler = False, use_one_hot_encoder = False)
            elif choice == '2':
                obj.prepare_data_to_model(dataset, percentage, use_standard_scaler = True, use_one_hot_encoder = False)
            elif choice == '3':
                obj.prepare_data_to_model(dataset, percentage, use_standard_scaler = False, use_one_hot_encoder = True)
            elif choice == '4':
                obj.prepare_data_to_model(dataset, percentage, use_standard_scaler = True, use_one_hot_encoder = True)
            elif choice == '5':
                loop=False
            else:
                input("Wrong option selection. Enter any key to try again..")
            
            menu4()
            return

def menu4():
    print ("\n" + 30 * "-" , "Choose one Algorithm" , 30 * "-")
    print ("1. Naive Bayes")
    print ("2. Decision Tree")
    print ("3. Random Forest")
    print ("4. kNN")
    print ("5. Exit")
    print (67 * "-")

    loop=True      
    while loop:        
            choice = input("Enter your choice [1-5]: ")
            
            if choice == '1':     
                obj.predict_naive_bayes()
            elif choice == '2':
                obj.predict_decision_tree()
            elif choice == '3':
                obj.predict_random_forest()
            elif choice == '4':
                obj.predict_knn()
            elif choice == '5':
                loop=False 
            else:
                input("Wrong option selection. Enter any key to try again..")
            return


obj = myClass() 
menu()