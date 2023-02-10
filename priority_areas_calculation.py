import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
import math

# Obtain settlement data
df = pd.read_csv('covid-colonias-mensual.csv')
df = df.dropna()

# Obtain settlement priorities defined by government
df_priority_colonies = pd.read_csv('colonias-de-atencion-prioritaria-covid-kioscos.csv')

# Separate the total population per settlement
df_population = df_priority_colonies[['clave_colonia','poblacion_total']]

# Separate the actual actives per settlement
df_actives = df_priority_colonies[['clave_colonia','activos']]

# Get the max quantity of actives
max_actives = df_priority_colonies[['activos']].sum()

# Add the total population to the settlement data
df_with_population = df.merge(df_population,left_on='clave_colonia',right_on='clave_colonia')

# Drop the values with NA
df_with_population = df_with_population.dropna()

# Get the value of maximum actives from pandas object
max_actives = max_actives[0]


# Group by settlements
gb = df_with_population.groupby('clave_colonia')

# Separate the data in different dataframes and sort each of them by date
colonies = [gb.get_group(x).sort_values(by='fecha_referencia') for x in gb.groups]

# Initialization of the processed data columns
slopes = np.array([])
colony_names = np.array([])
colony_id = np.array([])

# Function to plot the positive cases and the linear regression of the growth
def make_plots(x_labels, y, regression, title, file_path, x_label, y_label):
    plt.figure()
    plt.scatter(x_labels, y)
    plt.plot(x_labels, regression)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=30)
    plt.savefig(file_path)
    plt.close()   


# Iterate through the separated data
for colony in colonies:
    # Only use the settlements that have at least 2 reports of positive cases
    if len(colony.index) > 2:

        # Define labels of x axis of the graph (Dates)
        x_labels = colony['fecha_referencia'].tolist()

        # Make x axis for the regression (it is just a list of consecutive numbers)
        x = list(range(len(x_labels)))

        # Make list of positives to use with the x axis
        positives = colony['casos_positivos'].tolist()

        # Sum of positives at that time
        sum_positives = [sum(positives[:i+1]) for i,value in enumerate(positives)]
        
        # Percentage growth of the positives at each time (Sum of positives/Total population of the settlement)
        y = [x/colony['poblacion_total'].iloc[0] for x in sum_positives]

        # Debug prints😁
        #print(sum_positives, colony['poblacion_total'].iloc[0])
        #print(x)
        #print(positives)
        #print(sum_positives)

        # Linear regression of the percentage growth
        slope, intercept, r, p, std_err = stats.linregress(x, y)

        # Y values of the regression function
        regression = list(map(lambda x:slope*x+intercept,x))
        
        # Add values to each list: slope, settlement name, settlement id
        slopes = np.append(slopes,[slope])
        colony_names = np.append(colony_names,[colony['nombre_colonia'].iloc[0]])
        colony_id = np.append(colony_id,[colony['clave_colonia'].iloc[0]])

        # Delete the hashtag below to enable the creation of the plots
        #make_plots(x_labels,y,regression,colony['nombre_colonia'].iloc[0],f"colonies_pgrowth_and_approx/{colony['clave_colonia'].iloc[0]}.jpg","Fecha","Casos positivos")

# Total sum of the slopes
max_slopes = slopes.sum()

# Create the dataframe of all the saved columns
process_data = pd.DataFrame({'Growth': slopes, 'nombre_colonia': colony_names, 'clave_colonia': colony_id})

# Map the amount of the slope with the amount of total actives
actives_mapping = interp1d([0, max_slopes],[0, max_actives])

# Add the actives to the initial dataframe
process_data['optimized_actives'] = process_data.apply(lambda row: math.floor(actives_mapping(row['Growth'])), axis=1)

# Add the actives the government decided for each settlement
comparing_data = process_data.merge(df_actives,left_on='clave_colonia',right_on='clave_colonia')

# Add the difference between two quantities. The value is the amount needed to add to the government actives
comparing_data['actives_comparison'] = comparing_data.apply(lambda row: row['activos']-row['optimized_actives'], axis=1)

# Save the comparing data in a csv
comparing_data.to_csv("calculated_priorities.csv", index=False)

print(comparing_data)
print(process_data.sort_values(by='Growth', ascending=False))
print(max_slopes, max_actives)