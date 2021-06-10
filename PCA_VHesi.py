import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# use your own path to the data
#data_path = r'C:\Users\Jack Blumstein\STIP Summer Intership Stuff\covidcast-fb-survey-smoothed_covid_vaccinated_or_accept-2020-12-01-to-2021-06-07.csv'
data_path = r'C:\Users\Jack Blumstein\STIP Summer Intership Stuff\KNNImputedDataLessCounty.csv'

data = pd.read_csv(data_path)
# Read in the data, only use the time series data, drop the rest
# In the paper, we also removed counties that were in alaska, hawaii, and the territories but they are included in this data
# your results will likely be slightly different than in the paper because of this!
data = pd.read_csv(data_path, dtype={'geo_value':str}).set_index('geo_value')
data.columns = pd.to_datetime(data.columns)

# remove the 6 days of N/a's from taking the rolling average
#data = data.dropna(axis=1) 


# The number of components that we want to use
components = 2

# Initialize a model 
clf = PCA(n_components=components)

# Fit it to your data
clf.fit(data)

# Transform it to your data 
X = clf.transform(data)

# X = the data represented in PCA space. it 
print('Total Explained Var:',sum(clf.explained_variance_ratio_)) # total explained var of the model
print('Eiganvalues:',clf.explained_variance_ratio_) # show the eiganvalues for the PCs


# Create a dataframe from the model outputs
output = pd.DataFrame(X, index=data.index, columns=['PC_{}_Norm'.format(i) for i in range(1, components+1)])



def inverse_principal_components(df, model):
    '''
    Pass in the PCA model used and the output principal components. 

    This will return a dataframe with the principal components back in the original data space.
    '''
    # ineficient but works, isolates each principal component based on its max value in PCA space and the min for the rest 
    inverse = []
    for col in df.columns:
        temp = [] # holding list for each principal component maxes and mins
        for _col in df.columns:
            if col == _col:
                temp.append(df[_col].max())
            else:
                temp.append(df[_col].min())
        inverse.append(temp)

    # Create a dataframe from the inverse transformation that is back into the time series space and plot each of the principal components
    return pd.DataFrame(model.inverse_transform(inverse), columns=data.columns, index=['PC{}'.format(i) for i in range(1,components+1)])



inv_df = inverse_principal_components(output, clf)
inv_df.T.plot() # to plot these time series you have to transpose them 
plt.show()


# Normalize values if you want 0 to 1
output = (output - output.min())  / (output.max() - output.min())
print(output)